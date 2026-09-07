from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.build_index import IndexRow
from r2wsp.data.tcga_dataset import TCGAMultimodalDataset
from r2wsp.models.direct_survival import DirectWSIRNASurvival, local_geometry
from r2wsp.train.collate import collate_tiles_and_rna


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def neg_partial_log_likelihood(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    risk = risk[order].reshape(-1)
    events = events[order].reshape(-1)
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    diff = risk - log_cumsum
    denom = events.sum().clamp_min(1.0)
    return -(diff * events).sum() / denom


def window_weight(
    metric_value: float,
    *,
    lower: float,
    upper: float,
    k: float,
    A: float = 1.0,
) -> float:
    """Two-sided sigmoid window with amplitude A：

    f(x) = A * σ(k*(x - L)) * σ(-k*(x - U))

    A：幅度系数（灵敏度），默认 0.99 → 窗口内最大调节幅度≈0.99（Cox 最大削弱 1%）
    A=1：窗口内 f(x)≈1，只靠窗口门控开关，不额外缩放 Cox（“修改幅度不变”对照）
    窗口外 f≈0，后续再取 max(w, eps) 防梯度为 0。
    """
    if k <= 0:
        raise ValueError(f"window k must be positive, got {k}")
    if A <= 0:
        raise ValueError(f"window amplitude A must be > 0, got {A}")
    m = float(metric_value)
    left = float(1.0 / (1.0 + float(np.exp(-float(k) * (m - float(lower))))))
    right = float(1.0 / (1.0 + float(np.exp(float(k) * (m - float(upper))))))
    return float(A) * left * right


def window_weight_asym(
    metric_value: float,
    *,
    lower: float,
    upper: float,
    k_left: float,
    k_right: float,
    A: float = 1.0,
) -> float:
    if k_left <= 0 or k_right <= 0:
        raise ValueError(f"window k must be positive, got k_left={k_left}, k_right={k_right}")
    if A <= 0:
        raise ValueError(f"window amplitude A must be > 0, got {A}")
    m = float(metric_value)
    left = float(1.0 / (1.0 + float(np.exp(-float(k_left) * (m - float(lower))))))
    right = float(1.0 / (1.0 + float(np.exp(float(k_right) * (m - float(upper))))))
    return float(A) * left * right


def _ema_from_history(values: list[float], *, decay: float) -> float:
    if not values:
        return float("nan")
    ema = float(values[0])
    d = float(decay)
    for v in values[1:]:
        ema = d * ema + (1.0 - d) * float(v)
    return float(ema)


def _window_params_from_history(
    metric_history: list[float],
    *,
    policy: str,
    center_mode: str,
    width_mode: str,
    width_mult: float,
    width_quantile: float,
    width_min: float,
    width_max: float,
    transition: float,
    K: float,
    K_left: float | None,
    K_right: float | None,
    center_quantile: float,
    best_mix_alpha: float,
    fixed_lower: float,
    fixed_upper: float,
    fixed_k: float,
    ema_decay: float,
) -> tuple[float, float, float, float, float, float, float]:
    lower_fixed = float(fixed_lower)
    upper_fixed = float(fixed_upper)
    pol = str(policy)
    if pol not in ("param", "onek", "twok"):
        pol = "param"
    if not metric_history or (pol == "param" and str(center_mode) == "fixed" and str(width_mode) == "fixed" and float(transition) <= 0.0):
        width = float(max(upper_fixed - lower_fixed, 0.0))
        k = float((2.9444389791664403 / float(transition)) if float(transition) > 0.0 else float(fixed_k))
        center = float((lower_fixed + upper_fixed) * 0.5)
        return lower_fixed, upper_fixed, k, center, width, k, k

    history = [float(x) for x in metric_history if np.isfinite(float(x))]
    if not history:
        width = float(max(upper_fixed - lower_fixed, 0.0))
        k = float((2.9444389791664403 / float(transition)) if float(transition) > 0.0 else float(fixed_k))
        center = float((lower_fixed + upper_fixed) * 0.5)
        return lower_fixed, upper_fixed, k, center, width, k, k

    if pol in ("onek", "twok"):
        center = float(_ema_from_history(history, decay=float(ema_decay)))
        if len(history) >= 2:
            s = float(np.std(np.asarray(history, dtype=np.float64), ddof=0))
        else:
            s = float(max(0.5 * (upper_fixed - lower_fixed), 0.0))
        width = float(4.0 * s)
        width = float(min(max(width, 0.06), 0.60))
        lower = float(center - 0.5 * width)
        upper = float(center + 0.5 * width)
        lower = float(min(max(lower, 0.0), 1.0))
        upper = float(min(max(upper, 0.0), 1.0))
        if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
            lower = lower_fixed
            upper = upper_fixed
            width = float(max(upper_fixed - lower_fixed, 0.0))

        K_base = float(K)
        K_base = float(min(max(K_base, 0.02), 0.80))
        if pol == "onek":
            delta = float(max(K_base * width, 1e-6))
            k = float(2.9444389791664403 / delta)
            return float(lower), float(upper), float(k), float(center), float(width), float(k), float(k)

        Kl = float(K_left) if K_left is not None else K_base
        Kr = float(K_right) if K_right is not None else K_base
        Kl = float(min(max(Kl, 0.02), 0.80))
        Kr = float(min(max(Kr, 0.02), 0.80))
        delta_l = float(max(Kl * width, 1e-6))
        delta_r = float(max(Kr * width, 1e-6))
        k_left = float(2.9444389791664403 / delta_l)
        k_right = float(2.9444389791664403 / delta_r)
        k_sym = float(np.sqrt(float(k_left) * float(k_right)))
        return float(lower), float(upper), float(k_sym), float(center), float(width), float(k_left), float(k_right)

    cm = str(center_mode)
    if cm == "best":
        center = float(np.max(history))
    elif cm == "best_mix":
        a = float(best_mix_alpha)
        a = float(min(max(a, 0.0), 1.0))
        center_best = float(np.max(history))
        center_ema = float(_ema_from_history(history, decay=float(ema_decay)))
        center = float((1.0 - a) * center_best + a * center_ema)
    elif cm == "quantile":
        q = float(center_quantile)
        q = float(min(max(q, 0.5), 0.999))
        center = float(np.quantile(history, q))
    elif cm == "ema":
        center = float(_ema_from_history(history, decay=float(ema_decay)))
    else:
        center = float((lower_fixed + upper_fixed) * 0.5)

    wm = str(width_mode)
    if wm == "std" and len(history) >= 2:
        width = float(float(width_mult) * float(np.std(np.asarray(history, dtype=np.float64), ddof=0)))
    elif wm == "quantile" and len(history) >= 4:
        q = float(width_quantile)
        q = min(max(q, 0.5), 0.999)
        lo = float(np.quantile(history, (1.0 - q) * 0.5))
        hi = float(np.quantile(history, 1.0 - (1.0 - q) * 0.5))
        width = float(max(0.0, hi - lo))
    else:
        width = float(max(upper_fixed - lower_fixed, 0.0))

    width = float(min(max(width, float(width_min)), float(width_max)))
    k = float((2.9444389791664403 / float(transition)) if float(transition) > 0.0 else float(fixed_k))
    lower = float(center - 0.5 * width)
    upper = float(center + 0.5 * width)
    lower = float(min(max(lower, 0.0), 1.0))
    upper = float(min(max(upper, 0.0), 1.0))
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        lower = lower_fixed
        upper = upper_fixed
        width = float(max(upper_fixed - lower_fixed, 0.0))
    return float(lower), float(upper), float(k), float(center), float(width), float(k), float(k)


def _resolve_geo_points(aux: dict[str, torch.Tensor], branch: str) -> torch.Tensor | None:
    if str(branch) == "wsi":
        for key in ("wsi_curve_points", "wsi_b_points_points"):
            if key in aux:
                return aux[key]
    elif str(branch) == "rna":
        for key in ("rna_curve_points", "rna_b_points_points"):
            if key in aux:
                return aux[key]
    return None


def _geometry_regularizer(points: torch.Tensor | None, reg_type: str) -> torch.Tensor | None:
    if points is None or str(reg_type) == "none":
        return None
    if points.ndim != 3:
        raise ValueError(f"geometry regularizer expects points with shape [B, P, D], got {tuple(points.shape)}")
    if str(reg_type) == "laplacian_l2":
        if points.shape[1] < 3:
            return points.new_zeros(())
        lap = points[:, 2:, :] - 2.0 * points[:, 1:-1, :] + points[:, :-2, :]
        return lap.pow(2).sum(dim=-1).mean()
    if str(reg_type) == "velocity_l2":
        if points.shape[1] < 2:
            return points.new_zeros(())
        delta = points[:, 1:, :] - points[:, :-1, :]
        return delta.pow(2).sum(dim=-1).mean()
    if str(reg_type) == "tv_l1":
        if points.shape[1] < 2:
            return points.new_zeros(())
        delta = points[:, 1:, :] - points[:, :-1, :]
        return delta.abs().sum(dim=-1).mean()
    if str(reg_type) == "curvature_l2":
        curvature, _ = local_geometry(points)
        return curvature.pow(2).mean()
    if str(reg_type) == "torsion_l2":
        _, torsion = local_geometry(points)
        return torsion.pow(2).mean()
    raise ValueError(f"unsupported geometry regularizer: {reg_type}")


def _step_target_regularizer(points: torch.Tensor | None, target_step: float) -> torch.Tensor | None:
    if points is None or float(target_step) <= 0.0:
        return None
    if points.ndim != 3:
        raise ValueError(f"step regularizer expects points with shape [B, P, D], got {tuple(points.shape)}")
    if points.shape[1] < 2:
        return points.new_zeros(())
    delta = points[:, 1:, :] - points[:, :-1, :]
    step = torch.norm(delta, dim=-1)
    target = step.new_full(step.shape, float(target_step))
    return (step - target).pow(2).mean()


def compute_geometry_regularization(
    aux: dict[str, torch.Tensor],
    *,
    wsi_reg_type: str,
    wsi_reg_lambda: float,
    wsi_step_target: float,
    wsi_step_target_lambda: float,
    rna_reg_type: str,
    rna_reg_lambda: float,
    rna_step_target: float,
    rna_step_target_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(iter(aux.values())).device if aux else torch.device("cpu")
    total = torch.zeros((), device=device)
    details = {
        "wsi_geo_reg_unscaled": 0.0,
        "rna_geo_reg_unscaled": 0.0,
        "wsi_geo_reg_scaled": 0.0,
        "rna_geo_reg_scaled": 0.0,
        "wsi_geo_step_reg_unscaled": 0.0,
        "rna_geo_step_reg_unscaled": 0.0,
        "wsi_geo_step_reg_scaled": 0.0,
        "rna_geo_step_reg_scaled": 0.0,
    }
    for branch, reg_type, reg_lambda, step_target, step_target_lambda in (
        ("wsi", str(wsi_reg_type), float(wsi_reg_lambda), float(wsi_step_target), float(wsi_step_target_lambda)),
        ("rna", str(rna_reg_type), float(rna_reg_lambda), float(rna_step_target), float(rna_step_target_lambda)),
    ):
        points = _resolve_geo_points(aux, branch)
        if reg_type != "none" and reg_lambda > 0.0:
            reg = _geometry_regularizer(points, reg_type)
            if reg is not None:
                scaled = float(reg_lambda) * reg
                total = total + scaled
                details[f"{branch}_geo_reg_unscaled"] = float(reg.detach().cpu())
                details[f"{branch}_geo_reg_scaled"] = float(scaled.detach().cpu())
        if step_target_lambda > 0.0 and step_target > 0.0:
            step_reg = _step_target_regularizer(points, step_target)
            if step_reg is not None:
                step_scaled = float(step_target_lambda) * step_reg
                total = total + step_scaled
                details[f"{branch}_geo_step_reg_unscaled"] = float(step_reg.detach().cpu())
                details[f"{branch}_geo_step_reg_scaled"] = float(step_scaled.detach().cpu())
    return total, details


def concordance_index(risk: np.ndarray, times: np.ndarray, censorships: np.ndarray) -> float:
    events = 1.0 - censorships.astype(np.float32)
    concordant = 0.0
    comparable = 0.0
    n = len(risk)
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and events[i] > 0:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] > 0:
                comparable += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_official_split(split_dir: Path, target_col: str) -> tuple[dict[str, dict[str, float]], set[str], set[str]]:
    censorship_col = target_col.split("_")[0] + "_censorship"
    train_rows = read_split_csv(split_dir / "train.csv")
    test_rows = read_split_csv(split_dir / "test.csv")
    case_table: dict[str, dict[str, float]] = {}
    for row in train_rows + test_rows:
        case_id = str(row["case_id"])
        case_table[case_id] = {
            "event_time": float(row[target_col]),
            "censorship": float(row[censorship_col]),
        }
    return case_table, {str(r["case_id"]) for r in train_rows}, {str(r["case_id"]) for r in test_rows}


def _random_split_train_val_case_ids(train_case_ids: list[str], seed: int, val_frac: float) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(train_case_ids), dtype=object)
    rng.shuffle(ids)
    val_size = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(str(x) for x in ids[:val_size].tolist())
    train_ids = set(str(x) for x in ids[val_size:].tolist())
    return train_ids, val_ids


def _build_survival_strata(
    train_case_ids: list[str],
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int,
) -> dict[str, list[str]]:
    df = pd.DataFrame(
        {
            "case_id": [str(c) for c in train_case_ids],
            "event_time": [float(case_table[str(c)]["event_time"]) for c in train_case_ids],
            "censorship": [float(case_table[str(c)]["censorship"]) for c in train_case_ids],
        }
    )
    uncensored = df.loc[df["censorship"] < 0.5, "event_time"].to_numpy(dtype=np.float64)
    if uncensored.size >= 2 and n_time_bins > 1:
        quantiles = np.quantile(uncensored, np.linspace(0.0, 1.0, int(n_time_bins) + 1))
        quantiles[0] = min(quantiles[0], float(df["event_time"].min())) - 1e-6
        quantiles[-1] = max(quantiles[-1], float(df["event_time"].max())) + 1e-6
        quantiles = np.unique(quantiles)
        if quantiles.size >= 3:
            df["time_bin"] = pd.cut(
                df["event_time"],
                bins=quantiles,
                labels=False,
                include_lowest=True,
            ).fillna(0).astype(int)
        else:
            df["time_bin"] = 0
    else:
        df["time_bin"] = 0
    df["stratum"] = df["censorship"].astype(int).astype(str) + "_" + df["time_bin"].astype(int).astype(str)
    strata: dict[str, list[str]] = {}
    for stratum, sub_df in df.groupby("stratum", sort=True):
        strata[str(stratum)] = [str(x) for x in sub_df["case_id"].tolist()]
    return strata


def _survival_stratified_split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    strata = _build_survival_strata(train_case_ids, case_table, n_time_bins=n_time_bins)
    val_ids: list[str] = []
    train_ids: list[str] = []
    for stratum in sorted(strata.keys()):
        ids = np.asarray(sorted(strata[stratum]), dtype=object)
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.extend(str(x) for x in ids.tolist())
            continue
        val_size = int(round(len(ids) * float(val_frac)))
        val_size = max(1, min(len(ids) - 1, val_size))
        val_ids.extend(str(x) for x in ids[:val_size].tolist())
        train_ids.extend(str(x) for x in ids[val_size:].tolist())
    if not val_ids or not train_ids:
        return _random_split_train_val_case_ids(train_case_ids, seed, val_frac)
    return set(train_ids), set(val_ids)


def split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    mode: str,
    n_time_bins: int,
) -> tuple[set[str], set[str]]:
    if str(mode) == "survival_stratified":
        return _survival_stratified_split_train_val_case_ids(
            train_case_ids,
            seed,
            val_frac,
            case_table,
            n_time_bins=int(n_time_bins),
        )
    return _random_split_train_val_case_ids(train_case_ids, seed, val_frac)


def infer_cohort_from_split_dir(split_dir: Path) -> str | None:
    s = str(split_dir).upper()
    m = re.search(r"TCGA_([A-Z0-9]+)_OVERALL_SURVIVAL", s)
    if m is not None:
        return m.group(1)
    m = re.search(r"CENSORED_STAGE_PROTOCOL_([A-Z0-9]+)", s)
    if m is not None:
        return m.group(1)
    return None


def resolve_gene_sets_csv(raw_rna_root: Path, provided_path: str | None) -> Path:
    if provided_path is not None:
        path = Path(provided_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"rna gene sets csv not found: {path}")
        return path
    local_default = (raw_rna_root / "metadata" / "hallmarks_signatures.csv").resolve()
    if local_default.exists():
        return local_default
    raise FileNotFoundError(
        "No project-local hallmarks_signatures.csv found under data/raw_rna/metadata. "
        "Please pass --rna_gene_sets_csv explicitly."
    )


def strip_gene_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _parse_gtf_attributes(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(raw).strip().split(";"):
        item = chunk.strip()
        if not item or " " not in item:
            continue
        key, value = item.split(" ", 1)
        out[str(key)] = str(value).strip().strip('"')
    return out


def load_gene_id_to_symbol_map(gtf_path: Path) -> dict[str, str]:
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    mapping: dict[str, str] = {}
    with opener(gtf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = _parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            gene_name = attrs.get("gene_name")
            if not gene_id or not gene_name:
                continue
            mapping[strip_gene_version(gene_id)] = str(gene_name)
    if not mapping:
        raise ValueError(f"no gene_id -> gene_name mapping found in GTF: {gtf_path}")
    return mapping


def resolve_gene_annotation_gtf(provided_path: str | None) -> Path | None:
    if provided_path is not None:
        path = Path(provided_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"gene_annotation_gtf not found: {path}")
        return path
    auto_candidates = [
        Path("/root/autodl-tmp/gencode.v22.annotation.gtf.gz"),
        Path("/root/autodl-tmp/gencode.v22.annotation.gtf"),
    ]
    for candidate in auto_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def apply_cohort_hint(
    rows: list[IndexRow],
    *,
    cohort_hint: str | None,
    reference_rows: list[IndexRow] | None = None,
) -> list[IndexRow]:
    if cohort_hint is None:
        return rows
    cohort_key = str(cohort_hint).upper()
    ref_rows = reference_rows if reference_rows is not None else rows
    rna_path = next((r.rna_path for r in ref_rows if (r.cohort or "").upper() == cohort_key and r.rna_path is not None), None)
    clinical_path = next(
        (r.clinical_path for r in ref_rows if (r.cohort or "").upper() == cohort_key and r.clinical_path is not None),
        None,
    )
    if rna_path is None and clinical_path is None:
        try:
            paths = resolve_data_paths()
            candidate_rna = (paths.raw_rna_root / "tpm_tsv" / f"{cohort_key.lower()}_tpm.tsv").resolve()
            candidate_clin = (paths.raw_clinical_root / cohort_key / "clinical.csv").resolve()
            if candidate_rna.exists():
                try:
                    with open(candidate_rna, "r", encoding="utf-8", errors="ignore") as f:
                        header = f.readline().rstrip("\n")
                    cols = header.split("\t")
                    if len(cols) >= 2 and str(cols[1]).upper().startswith("TCGA-"):
                        rna_path = candidate_rna
                except Exception:
                    pass
            if candidate_clin.exists():
                clinical_path = candidate_clin
        except Exception:
            pass
    if rna_path is None and clinical_path is None:
        return rows
    repaired: list[IndexRow] = []
    for row in rows:
        if row.cohort is not None and row.rna_path is not None and row.clinical_path is not None:
            repaired.append(row)
            continue
        repaired.append(
            replace(
                row,
                cohort=row.cohort or cohort_key,
                rna_path=row.rna_path or rna_path,
                clinical_path=row.clinical_path or clinical_path,
            )
        )
    return repaired


def dedupe_rows_by_case(rows: list[IndexRow], *, source_name: str | None = None) -> list[IndexRow]:
    grouped: dict[str, list[IndexRow]] = defaultdict(list)
    for row in rows:
        if row.case_id is None:
            continue
        if source_name is not None and row.wsi_feature_source != source_name:
            continue
        grouped[str(row.case_id)].append(row)
    chosen: list[IndexRow] = []
    for case_id in sorted(grouped.keys()):
        items = sorted(grouped[case_id], key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))
        chosen.append(items[0])
    return chosen


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    train_c_index: float
    val_c_index: float
    test_c_index: float


def build_loader(
    rows: list[IndexRow],
    *,
    seed: int,
    batch_size: int,
    max_tiles: int | None,
    tile_sampling: str,
    tile_sampling_seed: int,
    rna_mode: str,
    gene_sets_csv: str | Path | None,
    gene_id_to_symbol: dict[str, str] | None,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    use_multi_slide: bool,
    multi_slide_tile_budget_mode: str,
    use_anti_features: bool,
    anti_feature_dir: str | Path | None,
    anti_feature_cohorts: list[str] | str | None,
    require_both_modalities: bool = True,
) -> DataLoader:
    ds = TCGAMultimodalDataset(
        rows,
        split=None,
        seed=seed,
        max_tiles=max_tiles,
        tile_sampling=tile_sampling,
        tile_sampling_seed=tile_sampling_seed,
        rna_mode=rna_mode,
        rna_gene_sets_csv=gene_sets_csv,
        rna_gene_id_to_symbol=gene_id_to_symbol,
        use_multi_slide=use_multi_slide,
        multi_slide_tile_budget_mode=multi_slide_tile_budget_mode,
        use_anti_features=bool(use_anti_features),
        anti_feature_dir=anti_feature_dir,
        anti_feature_cohorts=anti_feature_cohorts,
    )
    if require_both_modalities and len(rows) > 0:
        n_in = len({str(r.case_id) for r in rows if r.case_id})
        if use_multi_slide:
            out_cases = {str(r[0].case_id) for r in ds.rows if isinstance(r, list) and r}
        else:
            out_cases = {str(r.case_id) for r in ds.rows if r and not isinstance(r, list) and r.case_id}
        n_out = len(out_cases)
        if n_out == 0 or n_in == 0:
            raise ValueError(
                f"[require-both-modalities FAIL] dataset 构建后 case 全空：rows_in_n={n_in} ds_out_n={n_out}。"
                f" WSI feature_path 缺失或 RNA TSV 未覆盖，严禁单模态训。请检查 rows 头 5 条："
                f" {[(str(r.case_id), str(r.wsi_feature_path), str(r.rna_path)) for r in rows[:5]]}"
            )
        hit_rate = float(n_out) / float(n_in)
        if hit_rate < 0.9:
            missing_cases = sorted({str(r.case_id) for r in rows if r.case_id} - out_cases)
            raise ValueError(
                f"[require-both-modalities FAIL] 双模态 hit rate={hit_rate:.3f} < 0.9：rows_in_n={n_in} ds_out_n={n_out}。"
                f" 缺的 case={missing_cases[:20]}…"
            )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
        collate_fn=collate_tiles_and_rna,
    )


def gather_labels(case_ids: list[str], case_table: dict[str, dict[str, float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.tensor([case_table[c]["event_time"] for c in case_ids], dtype=torch.float32, device=device)
    censorships = torch.tensor([case_table[c]["censorship"] for c in case_ids], dtype=torch.float32, device=device)
    events = 1.0 - censorships
    return times, events


def evaluate(
    model: DirectWSIRNASurvival,
    loader: DataLoader,
    case_table: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    risks_t: list[torch.Tensor] = []
    times_t: list[torch.Tensor] = []
    events_t: list[torch.Tensor] = []
    times_np: list[np.ndarray] = []
    censorships_np: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_times_t, batch_events_t = gather_labels(batch.case_id, case_table, device)
            risk, _ = model(
                tile_tokens=batch.tile_tokens.to(device),
                tile_xy=batch.tile_xy.to(device),
                tile_attn_mask=batch.tile_attn_mask.to(device),
                slide_ids=batch.slide_ids.to(device),
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
                wsi_anti_vec=batch.wsi_anti_vec.to(device) if batch.wsi_anti_vec is not None else None,
                rna_anti_vec=batch.rna_anti_vec.to(device) if batch.rna_anti_vec is not None else None,
            )
            risks_t.append(risk.detach().cpu())
            times_t.append(batch_times_t.detach().cpu())
            events_t.append(batch_events_t.detach().cpu())
            times_np.append(batch_times_t.detach().cpu().numpy())
            censorships_np.append((1.0 - batch_events_t).detach().cpu().numpy())
    risk_all_t = torch.cat(risks_t, dim=0).reshape(-1)
    time_all_t = torch.cat(times_t, dim=0).reshape(-1)
    event_all_t = torch.cat(events_t, dim=0).reshape(-1)
    loss = neg_partial_log_likelihood(risk_all_t, time_all_t, event_all_t)

    risk_all = risk_all_t.numpy()
    time_all = np.concatenate(times_np, axis=0)
    censorship_all = np.concatenate(censorships_np, axis=0)
    return concordance_index(risk_all, time_all, censorship_all), float(loss.detach().cpu()), risk_all


def enforce_no_geo_contract(args) -> None:
    """严格 no-geo 基线：绝对不允许任何 geo 路径被静默触发。"""
    if str(args.wsi_geo_type) != "none":
        raise ValueError(
            f"[no-geo contract] this censored-stage script enforces strict no-geo baseline, got wsi_geo_type={args.wsi_geo_type}. "
            "Use wsi_geo_type=none explicitly."
        )
    if str(args.rna_geo_type) != "none":
        raise ValueError(
            f"[no-geo contract] this censored-stage script enforces strict no-geo baseline, got rna_geo_type={args.rna_geo_type}. "
            "Use rna_geo_type=none explicitly."
        )
    if str(args.model_variant) == "custom":
        raise ValueError(
            "[no-geo contract] this censored-stage script forbids model_variant=custom (it can execute geo silently). "
            "Use baseline/gate_only/aug_only/etc."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Censored-stage WSI+RNA survival: Phase-1 (pure uncensored) -> Phase-2 (uncensored+censored 5fold + windowed Cox). Strict no-geo."
    )
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--wsi_feature_source", default="plip_luad_256")
    p.add_argument("--wsi_feature_source_aliases", default=None, help="逗号分隔：把其他 source_name 视作等价的 wsi_feature_source（解决混装问题）。例：uni1024,uni1024_tilefix256_spatial")
    p.add_argument("--split_dir", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--max_tiles", type=int, default=4096)
    p.add_argument("--max_tiles_eval", type=int, default=None)
    p.add_argument("--tile_sampling", default="prefix", choices=["prefix", "random"])
    p.add_argument("--tile_sampling_eval", default=None, choices=["prefix", "random"])
    p.add_argument("--tile_sampling_seed", type=int, default=0)
    p.add_argument("--rna_mode", default="vec", choices=["vec", "omics", "omics_attn"])
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--lr_min", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--val_split_mode", default="survival_stratified", choices=["random", "survival_stratified"])
    p.add_argument("--val_time_bins", type=int, default=4)
    p.add_argument("--selection_metric", default="val_c_index_ema", choices=["val_c_index", "val_c_index_ema", "val_loss"])
    p.add_argument("--val_ema_decay", type=float, default=0.6)
    p.add_argument("--selection_min_epochs", type=int, default=8)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument(
        "--model_variant",
        default="baseline",
        choices=["baseline", "gate_only", "aug_only", "geo_only", "geo_gate", "geo_gate_v2", "geo_tile_attn", "geo_tile", "custom"],
    )
    p.add_argument("--fusion_mode", default="concat", choices=["concat", "gated"], help="Deprecated alias.")
    p.add_argument("--aug_dim", type=int, default=64)
    p.add_argument("--geo_dim", type=int, default=64)
    p.add_argument("--pool_method", default="attention", choices=["mean", "max", "attention", "gated_attention", "attention_sa"])
    p.add_argument("--gate_enabled", default="auto", choices=["auto", "true", "false"])
    p.add_argument(
        "--cross_modal_fusion",
        default="concat",
        choices=[
            "concat",
            "gated",
            "self_attn",
            "cross_attn_rna_queries_wsi",
            "cross_attn_wsi_queries_rna",
            "cross_attn_bidirectional",
            "cross_attn_self_attn",
            "across",
        ],
    )
    p.add_argument("--use_multi_slide", action="store_true")
    p.add_argument("--multi_slide_mode", default="slide_mean_case_attn", choices=["slide_mean_case_attn", "slide_attn_case_attn"])
    p.add_argument("--multi_slide_tile_budget_mode", default="per_slide", choices=["per_slide", "case_shared"])
    # geo 参数保留（和主线一致签名），但强制 none；这样模板命令和主线一致。
    p.add_argument("--wsi_geo_type", default="none", choices=["none", "curve", "b_points"])
    p.add_argument("--wsi_geo_num_points", type=int, default=24)
    p.add_argument("--wsi_geo_coord_dim", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--wsi_geo_output", default="points", choices=["points", "curvature", "enhanced", "all"])
    p.add_argument("--wsi_geo_reg_type", default="none", choices=["none", "laplacian_l2", "velocity_l2", "tv_l1", "curvature_l2", "torsion_l2"])
    p.add_argument("--wsi_geo_reg_lambda", type=float, default=0.0)
    p.add_argument("--wsi_geo_step_target", type=float, default=0.0)
    p.add_argument("--wsi_geo_step_target_lambda", type=float, default=0.0)
    p.add_argument("--wsi_geo_position", default="after_mean", choices=["after_mean", "before_mean", "parallel"])
    p.add_argument("--wsi_geo_fusion", default="concat", choices=["concat", "gated"])
    p.add_argument("--wsi_b_points_level", default="case", choices=["case", "slide"])
    p.add_argument("--rna_geo_type", default="none", choices=["none", "tile", "curve"])
    p.add_argument("--rna_geo_num_points", type=int, default=6)
    p.add_argument("--rna_geo_coord_dim", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--rna_geo_output", default="inherit", choices=["inherit", "points", "curvature", "enhanced", "all"])
    p.add_argument("--rna_geo_reg_type", default="none", choices=["none", "laplacian_l2", "velocity_l2", "tv_l1", "curvature_l2", "torsion_l2"])
    p.add_argument("--rna_geo_reg_lambda", type=float, default=0.0)
    p.add_argument("--rna_geo_step_target", type=float, default=0.0)
    p.add_argument("--rna_geo_step_target_lambda", type=float, default=0.0)
    p.add_argument("--rna_geo_position", default="after_rna_proj", choices=["after_rna_proj", "before_gate", "at_fused"])
    p.add_argument("--rna_geo_fusion", default="concat", choices=["concat", "multiply", "add"])
    p.add_argument("--geo_swap", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--cohort_hint", default=None, help="Override cohort hint for apply_cohort_hint (e.g. LUAD). When None, falls back to infer_cohort_from_split_dir.")
    p.add_argument("--out_dir", default="outputs/censored_stage_survival")

    # ---- Censored-stage 新增控制 ----
    p.add_argument("--stage", type=int, default=1, choices=[1, 2], help="1 = Phase-1 (pure uncensored, plain Cox); 2 = Phase-2 (censored included, windowed Cox + pretrain).")
    p.add_argument("--pretrain_checkpoint", type=str, default=None, help="stage=2 必传：Phase-1 对应 best.pt 的路径。")
    p.add_argument("--save_all_epochs", action="store_true", help="额外保存每个 epoch checkpoint（调试用）。")

    # ---- 窗口损失参数（stage=2 生效）----
    p.add_argument("--loss_window_lower", type=float, default=0.58, help="L：窗口下界。默认 0.68。如果要全区间 ablation：L=0 U=1。")
    p.add_argument("--loss_window_upper", type=float, default=0.65, help="U：窗口上界。默认 0.72。")
    p.add_argument("--loss_window_k", type=float, default=50.0, help="k：窗口陡峭系数（进出门的速度），越大越接近硬窗。默认 50。")
    p.add_argument("--loss_window_A", type=float, default=0.99, help="A：窗口幅度系数（灵敏度）。窗口内 f(x)最大值就是 A。默认 0.99（Cox 最大削弱 1%%）。A=1 则只靠门控，不额外缩放 Cox。")
    p.add_argument("--loss_window_eps", type=float, default=1e-3, help="窗口外最小权重（防止梯度完全归零导致训练停滞）。默认 1e-3。")
    p.add_argument("--loss_window_metric", default="val_c_index_ema", choices=["val_c_index", "val_c_index_ema"], help="用哪个指标触发窗口。默认 val_c_index_ema。")
    p.add_argument("--loss_window_policy", default="param", choices=["param", "onek", "twok"], help="窗口参数策略：param=按(L,U,k)与(center/width/transition)生成；onek=单K自动生成(L,U,k)；twok=左右不同K自动生成。")
    p.add_argument("--loss_window_K", type=float, default=0.25, help="onek/twok：单K（或默认K）的转场比例，k≈2.944/(K*width)。默认 0.25。")
    p.add_argument("--loss_window_K_left", type=float, default=None, help="twok：左边界K（为空则用 --loss_window_K）。")
    p.add_argument("--loss_window_K_right", type=float, default=None, help="twok：右边界K（为空则用 --loss_window_K）。")
    p.add_argument("--loss_window_center_mode", default="fixed", choices=["fixed", "best", "best_mix", "quantile", "ema"], help="窗口中心 c 的来源（仅 policy=param 生效）：fixed=使用(L,U)固定中心；best=历史最佳；best_mix=best与EMA混合；quantile=历史分位数；ema=历史EMA。")
    p.add_argument("--loss_window_center_quantile", type=float, default=0.90, help="center_mode=quantile 时的分位数 q（0.5~0.999）。默认 0.90。")
    p.add_argument("--loss_window_best_mix_alpha", type=float, default=0.50, help="center_mode=best_mix 的混合系数 α：c=(1-α)*best+α*EMA。默认 0.50。")
    p.add_argument("--loss_window_width_mode", default="fixed", choices=["fixed", "std", "quantile"], help="窗口宽度 w 的来源：fixed=U-L；std=width_mult*std(history)；quantile=history 的中心分位宽度。")
    p.add_argument("--loss_window_width_mult", type=float, default=2.0, help="width_mode=std 时的倍数系数。")
    p.add_argument("--loss_window_width_quantile", type=float, default=0.8, help="width_mode=quantile 时的中心分位宽度（取 (1-q)/2 到 1-(1-q)/2 的分位差）。")
    p.add_argument("--loss_window_width_min", type=float, default=0.02, help="自适应窗口最小宽度下限。")
    p.add_argument("--loss_window_width_max", type=float, default=0.30, help="自适应窗口最大宽度上限。")
    p.add_argument("--loss_window_transition", type=float, default=0.0, help="用转场宽度 δ 反推 k（0.05→0.95 约跨 δ）：k≈2.944/δ。设为 0 则使用 --loss_window_k。")
    p.add_argument("--pullback_lambda", type=float, default=1e-4, help="Stage-2 回拉正则强度 λ：||θ-θ_stage1||^2 的系数。默认 1e-4。")

    # ---- Anti-feature injection (方案A: 冻结CPU预计算反特征做 gated residual add) ----
    p.add_argument("--use-anti-injection", action="store_true", help="启用方案A：把 outputs/anti_feature_experiment 里预计算的 z_wsi_*.npy / z_rna_*.npy (128-d) 冻结后 gated residual 加到 wsi_case/rna_case 里。旧 checkpoint strict=False 加载，无兼容性问题。")
    p.add_argument("--anti-feature-dir", default="outputs/anti_feature_experiment", help="Anti-feature 预计算 npy 目录。结构：z_wsi_{cohort}.npy / z_rna_{cohort}.npy / sample_ids_{cohort}.txt")
    p.add_argument("--anti-feature-cohorts", default=None, help="显式指定要加载 anti-feature 的 cohort 列表，逗号分隔。默认=自动从 rows 中推断所有 cohort。")
    p.add_argument("--anti-dim", type=int, default=128, help="当 anti-feature npy 未找到 fallback 时的默认维度。")
    p.add_argument("--require-both-modalities", default="true", choices=["true", "false"],
                   help="默认=true：任何 row 缺 RNA TSV hit 或 WSI feature 文件时，直接在 dataset 构建阶段抛错，严禁单模态训。调试时=false 允许单模态。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    enforce_no_geo_contract(args)

    set_seed(int(args.seed))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)

    stage = int(args.stage)
    pretrain_summary: dict | None = None
    pullback_base: dict[str, torch.Tensor] | None = None
    if stage == 2 and not args.pretrain_checkpoint:
        raise ValueError("stage=2 必须传 --pretrain_checkpoint（Phase-1 best.pt）。")

    model_variant = str(args.model_variant)
    if model_variant == "baseline" and str(args.fusion_mode) == "gated":
        model_variant = "gate_only"
    gate_enabled = None
    if str(args.gate_enabled) == "true":
        gate_enabled = True
    elif str(args.gate_enabled) == "false":
        gate_enabled = False

    # 主线里 non-custom + rna_geo/wsi_curve 也有护栏，这里重复一次防止配置漂移
    if model_variant != "custom":
        if str(args.rna_geo_type) != "none":
            raise ValueError("rna_geo_type is only executed when model_variant=custom")
        if str(args.rna_geo_output) != "inherit":
            raise ValueError("rna_geo_output is only executed when model_variant=custom")
        if str(args.wsi_geo_type) == "curve":
            raise ValueError("wsi_geo_type=curve is only executed when model_variant=custom")

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()
    dataset_rna_mode = "omics" if str(args.rna_mode) in {"omics", "omics_attn"} else "vec"
    gene_sets_csv = None
    gene_id_to_symbol = None
    gtf_path = None
    if dataset_rna_mode == "omics":
        gene_sets_csv = resolve_gene_sets_csv(paths.raw_rna_root, args.rna_gene_sets_csv)
        gtf_path = resolve_gene_annotation_gtf(args.gene_annotation_gtf)
        if gtf_path is not None:
            gene_id_to_symbol = load_gene_id_to_symbol_map(gtf_path)
        else:
            print("[warn] no gene_annotation_gtf found, omics matching will use raw gene ids")

    inv = scan_all_assets(paths)
    all_rows = build_index(inv)
    split_dir = Path(args.split_dir).resolve()
    cohort_hint = args.cohort_hint if args.cohort_hint else infer_cohort_from_split_dir(split_dir)
    # --- wsi_feature_source alias resolution（解决 uni1024 / uni1024_tilefix256_spatial 混装问题）---
    primary_source = str(args.wsi_feature_source)
    accepted_sources = {primary_source}
    if args.wsi_feature_source_aliases:
        for alias in str(args.wsi_feature_source_aliases).split(","):
            alias = alias.strip()
            if alias:
                accepted_sources.add(alias)
    # 把所有别名的 IndexRow 的 wsi_feature_source 统一重写为 primary_source，后续所有 source_name 过滤、dedupe、checkpoint 均一致
    if len(accepted_sources) > 1:
        alias_rewrite = 0
        for i, r in enumerate(all_rows):
            if str(r.wsi_feature_source) in accepted_sources and str(r.wsi_feature_source) != primary_source:
                all_rows[i] = IndexRow(
                    cohort=r.cohort, case_id=r.case_id, slide_id=r.slide_id,
                    svs_path=r.svs_path,
                    wsi_feature_path=r.wsi_feature_path,
                    wsi_feature_source=primary_source,
                    wsi_feature_kind=r.wsi_feature_kind,
                    rna_path=r.rna_path,
                    clinical_path=r.clinical_path,
                )
                alias_rewrite += 1
        if alias_rewrite:
            print(f"[INFO] wsi_feature_source alias rewrite: {alias_rewrite} rows from {sorted(accepted_sources - {primary_source})} -> {primary_source}")
    rows = apply_cohort_hint(
        all_rows if bool(args.use_multi_slide) else dedupe_rows_by_case(all_rows, source_name=primary_source),
        cohort_hint=cohort_hint,
        reference_rows=all_rows,
    )
    rows = [r for r in rows if str(r.wsi_feature_source) == primary_source]
    # --- cohort_hint 强制执行：如果某个 cohort/rna_path/clinical_path 仍为空（WSI token 没有匹配 svs 时会出现），按 cohort_hint 回填
    # 注意：IndexRow 里的 cohort 是 asset scan 时按路径推断的，大多 WSI 路径不含 cohort 名，所以需要显式按 split_dir 的数据归属填。
    if cohort_hint:
        cohort_key = str(cohort_hint).upper()
        rna_path_fill, clinical_path_fill = None, None
        # --- 直接从 DataPaths 扫描 clinical/rna asset，走 scan_clinical_assets + scan_rna_assets，不再依赖 WSI IndexRow (cohort 常为 None)
        from r2wsp.data.paths import resolve_data_paths as _resolve
        from r2wsp.data.scan_assets import scan_clinical_assets, scan_rna_assets as _scan_rna
        _paths = paths
        # scan_clinical: raw_clinical/<COHORT>/clinical.csv → cohort=COHORT.upper() (直接匹配)
        _cli_assets = scan_clinical_assets(_paths.raw_clinical_root)
        for c in _cli_assets:
            if (c.cohort or "").upper() == cohort_key:
                clinical_path_fill = c.path
                break
        # scan_rna: source_dir = tpm_tsv/...，infer_cohort(filename) 已命中 brca/brca_symbol_tpm 等
        _rna_assets = _scan_rna(_paths.raw_rna_root)
        # 优先级：样本列最多的 TSV（大矩阵覆盖更多 case_id）> source=tpm_tsv > 带 symbol 关键词
        # 说明：ENSG TSV (brca_tpm.tsv) 列数通常为 1200+（完整 GDC 下载），symbol TSV 仅 336 列（补了一部分）。
        #       build_omics_spec 内部会通过 gene_id_to_symbol 把 ENSG ID 转成 HGNC symbol（gtf 存在），完美对齐 Hallmark50。
        candidates = []
        for r in _rna_assets:
            if (r.cohort or "").upper() == cohort_key and r.file_type == "tsv":
                try:
                    cols = __import__("pandas").read_csv(r.path, sep="\t", nrows=0).columns.tolist()
                    if len(cols) < 3:
                        ncol = 0
                    else:
                        c1 = str(cols[1]).upper()
                        if c1.startswith("TCGA-"):
                            ncol = len(cols) - 1
                        else:
                            ncol = 0
                except Exception:
                    ncol = 0
                score = ncol
                if r.source_name.lower() == "tpm_tsv":
                    score += 50
                candidates.append((score, ncol, Path(r.path).name, r.path))
        candidates.sort(reverse=True)
        if candidates:
            _, ncol, fname, rna_path_fill = candidates[0]
            print(f"[INFO] RNA TSV pick for {cohort_key}: {fname} (samples={ncol}, score={candidates[0][0]}, "
                  f"candidates={[(c[2], c[1]) for c in candidates[:3]]})")
        if rna_path_fill is None or clinical_path_fill is None:
            _rna_avail = sorted({(str(r.cohort), r.source_name, r.file_type, Path(r.path).name) for r in _rna_assets if r.cohort})[:15]
            _cli_avail = sorted({(str(c.cohort), Path(c.path).parent.name, Path(c.path).name) for c in _cli_assets})
            raise SystemExit(
                f"[ERROR] cohort_hint={cohort_hint} 未从 DataPaths 资产扫描中命中对应 RNA/clinical。\n"
                f"  Clinical cohorts 可用: {_cli_avail[:10]}\n"
                f"  RNA TSVs with cohort (top): {_rna_avail}\n"
                f"  当前 resolve 的 raw_rna_root={_paths.raw_rna_root}\n"
                f"  当前 resolve 的 raw_clinical_root={_paths.raw_clinical_root}"
            )
        fixed = 0
        for i, r in enumerate(rows):
            if (r.cohort is None or r.rna_path is None or r.clinical_path is None):
                rows[i] = replace(
                    r,
                    cohort=r.cohort or cohort_key,
                    rna_path=r.rna_path or rna_path_fill,
                    clinical_path=r.clinical_path or clinical_path_fill,
                )
                fixed += 1
        if fixed:
            print(f"[INFO] cohort_hint fallback repair: fixed {fixed} rows → cohort={cohort_key}, rna={Path(str(rna_path_fill)).name}, cli={Path(str(clinical_path_fill)).parent.name}/{Path(str(clinical_path_fill)).name}")



    case_table, train_case_ids, test_case_ids = load_official_split(split_dir, str(args.target_col))
    rows_case_ids = {str(r.case_id) for r in rows if r.case_id}
    train_case_ids = [c for c in train_case_ids if c in rows_case_ids and c in case_table]
    test_case_ids = [c for c in test_case_ids if c in rows_case_ids and c in case_table]
    if not train_case_ids or not test_case_ids:
        raise ValueError(f"[ERROR] split={split_dir} 经 rows 过滤后为空！train={len(train_case_ids)} test={len(test_case_ids)}。rows_case_ids={len(rows_case_ids)}, case_table={len(case_table)}")
    # --- 过滤 case_ids 到 rna TSV 实际覆盖的范围（极少数 WSI 有但 RNA 完全下不到的 case 会被剔除，BRCA 发现 1/318 = 0.3%）
    rows_by_case: dict[str, list[IndexRow]] = {}
    for r in rows:
        if r.case_id:
            rows_by_case.setdefault(str(r.case_id), []).append(r)
    tsv_3seg_coverage_per_path: dict[str, set[str]] = {}
    for c in list(train_case_ids) + list(test_case_ids):
        first_r = rows_by_case[c][0]
        rna_p = first_r.rna_path
        if rna_p is None:
            continue
        key = str(rna_p)
        if key not in tsv_3seg_coverage_per_path:
            tsv_3seg_coverage_per_path[key] = set()
            try:
                with open(Path(rna_p), "r", encoding="utf-8", errors="ignore") as f:
                    header = f.readline().rstrip("\n")
                cols = header.split("\t")
                for col_s in cols:
                    if col_s.startswith("TCGA-"):
                        tsv_3seg_coverage_per_path[key].add(col_s[:12])
                        tsv_3seg_coverage_per_path[key].add(col_s)
            except Exception as e:
                print(f"[WARN] TPM TSV header parse failed: path={rna_p} err={type(e).__name__}: {e}")
    def _in_tsv(case_id: str) -> bool:
        first_r = rows_by_case.get(case_id)
        if not first_r:
            return False
        key = str(first_r[0].rna_path)
        cov = tsv_3seg_coverage_per_path.get(key, set())
        return case_id in cov or case_id[:12] in cov
    before_train, before_test = len(train_case_ids), len(test_case_ids)
    drop_tr = [c for c in train_case_ids if not _in_tsv(c)]
    drop_te = [c for c in test_case_ids if not _in_tsv(c)]
    train_case_ids = [c for c in train_case_ids if _in_tsv(c)]
    test_case_ids = [c for c in test_case_ids if _in_tsv(c)]
    if drop_tr or drop_te:
        print(f"[INFO] TPM TSV 覆盖过滤：train {before_train}→{len(train_case_ids)}（drop {len(drop_tr)}：{drop_tr}），test {before_test}→{len(test_case_ids)}（drop {len(drop_te)}：{drop_te}）")
    if not train_case_ids or not test_case_ids:
        raise ValueError(f"[ERROR] TPM TSV 过滤后 train/test 为空！split={split_dir}")
    train_case_ids, val_case_ids = split_train_val_case_ids(
        sorted(train_case_ids),
        int(args.seed),
        float(args.val_frac),
        case_table,
        mode=str(args.val_split_mode),
        n_time_bins=int(args.val_time_bins),
    )

    train_rows = [r for r in rows if str(r.case_id) in train_case_ids]
    val_rows = [r for r in rows if str(r.case_id) in val_case_ids]
    test_rows = [r for r in rows if str(r.case_id) in test_case_ids]

    max_tiles_train: int | None = None if int(args.max_tiles) <= 0 else int(args.max_tiles)
    max_tiles_eval_raw = args.max_tiles if args.max_tiles_eval is None else args.max_tiles_eval
    max_tiles_eval: int | None = None if int(max_tiles_eval_raw) <= 0 else int(max_tiles_eval_raw)
    tile_sampling_train = str(args.tile_sampling)
    tile_sampling_eval = str(args.tile_sampling_eval) if args.tile_sampling_eval is not None else tile_sampling_train

    build_kwargs_base = dict(
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        tile_sampling_seed=int(args.tile_sampling_seed),
        rna_mode=dataset_rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
        use_anti_features=bool(args.use_anti_injection),
        anti_feature_dir=args.anti_feature_dir,
        anti_feature_cohorts=(
            [c.strip() for c in args.anti_feature_cohorts.split(",") if c.strip()]
            if args.anti_feature_cohorts else None
        ),
        require_both_modalities=str(args.require_both_modalities).lower() == "true",
    )
    build_kwargs_train = dict(build_kwargs_base, max_tiles=max_tiles_train, tile_sampling=tile_sampling_train)
    build_kwargs_eval = dict(build_kwargs_base, max_tiles=max_tiles_eval, tile_sampling=tile_sampling_eval)

    train_loader = build_loader(train_rows, shuffle=True, **build_kwargs_train)
    train_eval_loader = build_loader(train_rows, shuffle=False, **build_kwargs_eval)
    val_loader = build_loader(val_rows, shuffle=False, **build_kwargs_eval)
    test_loader = build_loader(test_rows, shuffle=False, **build_kwargs_eval)

    sample_batch = next(iter(train_loader))
    tile_dim = int(sample_batch.tile_tokens.shape[-1])
    rna_dim = int(sample_batch.rna_vec.shape[-1]) if sample_batch.rna_vec is not None else 1
    rna_omic_sizes = [int(x.shape[-1]) for x in sample_batch.rna_omics] if sample_batch.rna_omics is not None else None
    anti_dim = (
        int(sample_batch.wsi_anti_vec.shape[-1])
        if sample_batch.wsi_anti_vec is not None and sample_batch.rna_anti_vec is not None
        else int(getattr(args, "anti_dim", 128))
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))
    model = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        model_variant=model_variant,
        aug_dim=int(args.aug_dim),
        geo_dim=int(args.geo_dim),
        pool_method=str(args.pool_method),
        gate_enabled=gate_enabled,
        wsi_geo_type=str(args.wsi_geo_type),
        wsi_geo_num_points=int(args.wsi_geo_num_points),
        wsi_geo_coord_dim=int(args.wsi_geo_coord_dim),
        wsi_geo_output=str(args.wsi_geo_output),
        wsi_geo_position=str(args.wsi_geo_position),
        wsi_geo_fusion=str(args.wsi_geo_fusion),
        wsi_b_points_level=str(args.wsi_b_points_level),
        rna_geo_type=str(args.rna_geo_type),
        rna_geo_num_points=int(args.rna_geo_num_points),
        rna_geo_coord_dim=int(args.rna_geo_coord_dim),
        rna_geo_output=str(args.rna_geo_output),
        rna_geo_position=str(args.rna_geo_position),
        rna_geo_fusion=str(args.rna_geo_fusion),
        geo_swap=bool(args.geo_swap),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_mode=str(args.multi_slide_mode),
        rna_mode=str(args.rna_mode),
        rna_omic_sizes=rna_omic_sizes,
        cross_modal_fusion=str(args.cross_modal_fusion),
        anti_dim=anti_dim,
        use_anti_injection=bool(args.use_anti_injection),
    ).to(device)

    if stage == 2:
        ckpt_path = Path(args.pretrain_checkpoint).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"pretrain checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        pullback_base = {k: v.to(device) for k, v in state.items()}
        pretrain_summary = {
            "pretrain_source": str(ckpt_path),
            "pretrain_best_epoch": int(ckpt.get("best_epoch", -1)),
            "pretrain_best_val_c_index": float(ckpt.get("best_val_c_index", float("nan"))),
            "pretrain_best_test_c_index": float(ckpt.get("best_test_c_index", float("nan"))),
            "load_state_missing": list(missing),
            "load_state_unexpected": list(unexpected),
        }
        print(
            f"[stage2] loaded pretrain best.pt: {ckpt_path} "
            f"best_epoch={pretrain_summary['pretrain_best_epoch']} "
            f"best_test={pretrain_summary['pretrain_best_test_c_index']:.4f}"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(args.epochs),
            eta_min=float(args.lr_min),
        )

    history_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_val = float("-inf")
    best_test = float("nan")
    best_train = float("nan")
    best_val_loss = float("inf")
    best_select_score = float("-inf") if str(args.selection_metric) != "val_loss" else float("inf")
    val_c_index_ema: float | None = None
    window_metric_history: list[float] = []
    stale_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_task_losses: list[float] = []
        train_geo_reg_losses: list[float] = []
        train_wsi_geo_reg_losses: list[float] = []
        train_rna_geo_reg_losses: list[float] = []
        train_wsi_geo_step_reg_losses: list[float] = []
        train_rna_geo_step_reg_losses: list[float] = []
        epoch_window_weights: list[float] = []
        train_pullback_l2: list[float] = []
        train_pullback_losses: list[float] = []

        # --- Censored-stage window weight 计算：避免未来信息泄漏 ---
        # epoch 1：没有 val 指标信息 → w = 1.0（正常 Cox）
        # epoch>1 & stage=2：用上一 epoch 结束时的 {val_c_index / val_c_index_ema} 算窗口
        # → loss = w * (task_loss + geo_reg)  （geo_reg 在 no-geo 合约下恒为 0）
        # → w = max( window_weight(m_prev, A=A, L, U, k), eps )
        if epoch == 1 or stage == 1:
            epoch_window_weight = 1.0
        else:
            metric_prev = float(window_metric_history[-1]) if window_metric_history else float("nan")
            if np.isfinite(metric_prev):
                lower, upper, k, _, _, k_left, k_right = _window_params_from_history(
                    window_metric_history,
                    policy=str(args.loss_window_policy),
                    center_mode=str(args.loss_window_center_mode),
                    width_mode=str(args.loss_window_width_mode),
                    width_mult=float(args.loss_window_width_mult),
                    width_quantile=float(args.loss_window_width_quantile),
                    width_min=float(args.loss_window_width_min),
                    width_max=float(args.loss_window_width_max),
                    transition=float(args.loss_window_transition),
                    K=float(args.loss_window_K),
                    K_left=None if args.loss_window_K_left is None else float(args.loss_window_K_left),
                    K_right=None if args.loss_window_K_right is None else float(args.loss_window_K_right),
                    center_quantile=float(args.loss_window_center_quantile),
                    best_mix_alpha=float(args.loss_window_best_mix_alpha),
                    fixed_lower=float(args.loss_window_lower),
                    fixed_upper=float(args.loss_window_upper),
                    fixed_k=float(args.loss_window_k),
                    ema_decay=float(args.val_ema_decay),
                )
                A_eff = 1.0 if str(args.loss_window_policy) in ("onek", "twok") else float(args.loss_window_A)
                if str(args.loss_window_policy) == "twok":
                    w = window_weight_asym(metric_prev, lower=lower, upper=upper, k_left=k_left, k_right=k_right, A=A_eff)
                else:
                    w = window_weight(metric_prev, lower=lower, upper=upper, k=k, A=A_eff)
                epoch_window_weight = float(max(w, float(args.loss_window_eps)))
            else:
                epoch_window_weight = float(args.loss_window_eps)

        for batch in train_loader:
            times, events = gather_labels(batch.case_id, case_table, device)
            risk, aux = model(
                tile_tokens=batch.tile_tokens.to(device),
                tile_xy=batch.tile_xy.to(device),
                tile_attn_mask=batch.tile_attn_mask.to(device),
                slide_ids=batch.slide_ids.to(device),
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
                wsi_anti_vec=batch.wsi_anti_vec.to(device) if batch.wsi_anti_vec is not None else None,
                rna_anti_vec=batch.rna_anti_vec.to(device) if batch.rna_anti_vec is not None else None,
            )
            task_loss = neg_partial_log_likelihood(risk, times, events)
            geo_reg_loss, geo_reg_details = compute_geometry_regularization(
                aux,
                wsi_reg_type=str(args.wsi_geo_reg_type),
                wsi_reg_lambda=float(args.wsi_geo_reg_lambda),
                wsi_step_target=float(args.wsi_geo_step_target),
                wsi_step_target_lambda=float(args.wsi_geo_step_target_lambda),
                rna_reg_type=str(args.rna_geo_reg_type),
                rna_reg_lambda=float(args.rna_geo_reg_lambda),
                rna_step_target=float(args.rna_geo_step_target),
                rna_step_target_lambda=float(args.rna_geo_step_target_lambda),
            )
            raw_loss = task_loss + geo_reg_loss
            pull_l2 = torch.zeros((), device=device)
            if stage == 2 and pullback_base is not None and float(args.pullback_lambda) > 0.0:
                for name, p in model.named_parameters():
                    ref = pullback_base.get(name)
                    if ref is None or ref.shape != p.shape:
                        continue
                    pull_l2 = pull_l2 + (p - ref).pow(2).sum()
            pull_loss = float(args.pullback_lambda) * pull_l2
            loss = float(epoch_window_weight) * raw_loss + (1.0 - float(epoch_window_weight)) * pull_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_task_losses.append(float(task_loss.detach().cpu()))
            train_geo_reg_losses.append(float(geo_reg_loss.detach().cpu()))
            train_wsi_geo_reg_losses.append(float(geo_reg_details["wsi_geo_reg_scaled"]))
            train_rna_geo_reg_losses.append(float(geo_reg_details["rna_geo_reg_scaled"]))
            train_wsi_geo_step_reg_losses.append(float(geo_reg_details["wsi_geo_step_reg_scaled"]))
            train_rna_geo_step_reg_losses.append(float(geo_reg_details["rna_geo_step_reg_scaled"]))
            epoch_window_weights.append(float(epoch_window_weight))
            train_pullback_l2.append(float(pull_l2.detach().cpu()))
            train_pullback_losses.append(float(pull_loss.detach().cpu()))

        train_c_index, train_loss_eval, _ = evaluate(model, train_eval_loader, case_table, device)
        val_c_index, val_loss, _ = evaluate(model, val_loader, case_table, device)
        test_c_index, _, _ = evaluate(model, test_loader, case_table, device)
        if val_c_index_ema is None:
            val_c_index_ema = float(val_c_index)
        else:
            decay = float(args.val_ema_decay)
            val_c_index_ema = decay * float(val_c_index_ema) + (1.0 - decay) * float(val_c_index)

        if str(args.selection_metric) == "val_loss":
            select_score = float(val_loss)
            improved = (
                epoch >= int(args.selection_min_epochs)
                and (select_score < best_select_score if np.isfinite(best_select_score) else True)
            )
        elif str(args.selection_metric) == "val_c_index_ema":
            select_score = float(val_c_index_ema)
            improved = epoch >= int(args.selection_min_epochs) and select_score >= float(best_select_score)
        else:
            select_score = float(val_c_index)
            improved = epoch >= int(args.selection_min_epochs) and select_score >= float(best_select_score)

        if improved:
            best_val = float(val_c_index)
            best_val_loss = float(val_loss)
            best_test = float(test_c_index)
            best_train = float(train_c_index)
            best_epoch = epoch
            best_select_score = float(select_score)
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_c_index": best_val,
                    "best_val_loss": best_val_loss,
                    "best_test_c_index": best_test,
                    "best_select_score": best_select_score,
                    "stage": stage,
                    "pretrain": pretrain_summary,
                },
                out_dir / "best.pt",
            )
        else:
            stale_epochs += 1

        # --- Stage-2：每轮覆盖写 final.pt（不做 best 选择），符合“最后一轮直接检测” ---
        if stage == 2:
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "last_epoch": epoch,
                    "last_val_c_index": float(val_c_index),
                    "last_val_c_index_ema": float(val_c_index_ema),
                    "last_test_c_index": float(test_c_index),
                    "stage": stage,
                    "pretrain": pretrain_summary,
                },
                out_dir / "final.pt",
            )

        if bool(args.save_all_epochs):
            p = out_dir / "all_epochs" / f"epoch_{epoch}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch}, p)

        window_lower_now = float("nan")
        window_upper_now = float("nan")
        window_k_now = float("nan")
        window_k_left_now = float("nan")
        window_k_right_now = float("nan")
        window_center_now = float("nan")
        window_width_now = float("nan")
        if stage == 2:
            window_metric_now = float(val_c_index_ema) if args.loss_window_metric == "val_c_index_ema" else float(val_c_index)
            window_metric_history.append(float(window_metric_now))
            (
                window_lower_now,
                window_upper_now,
                window_k_now,
                window_center_now,
                window_width_now,
                window_k_left_now,
                window_k_right_now,
            ) = _window_params_from_history(
                window_metric_history,
                policy=str(args.loss_window_policy),
                center_mode=str(args.loss_window_center_mode),
                width_mode=str(args.loss_window_width_mode),
                width_mult=float(args.loss_window_width_mult),
                width_quantile=float(args.loss_window_width_quantile),
                width_min=float(args.loss_window_width_min),
                width_max=float(args.loss_window_width_max),
                transition=float(args.loss_window_transition),
                K=float(args.loss_window_K),
                K_left=None if args.loss_window_K_left is None else float(args.loss_window_K_left),
                K_right=None if args.loss_window_K_right is None else float(args.loss_window_K_right),
                center_quantile=float(args.loss_window_center_quantile),
                best_mix_alpha=float(args.loss_window_best_mix_alpha),
                fixed_lower=float(args.loss_window_lower),
                fixed_upper=float(args.loss_window_upper),
                fixed_k=float(args.loss_window_k),
                ema_decay=float(args.val_ema_decay),
            )
            A_eff = 1.0 if str(args.loss_window_policy) in ("onek", "twok") else float(args.loss_window_A)
            if str(args.loss_window_policy) == "twok":
                w_now = window_weight_asym(
                    window_metric_now,
                    lower=float(window_lower_now),
                    upper=float(window_upper_now),
                    k_left=float(window_k_left_now),
                    k_right=float(window_k_right_now),
                    A=float(A_eff),
                )
            else:
                w_now = window_weight(
                    window_metric_now,
                    lower=float(window_lower_now),
                    upper=float(window_upper_now),
                    k=float(window_k_now),
                    A=float(A_eff),
                )
        else:
            window_metric_now = float("nan")
            w_now = float("nan")

        row = {
            "stage": stage,
            "epoch": epoch,
            "loss": float(np.mean(train_losses)),
            "task_loss": float(np.mean(train_task_losses)),
            "geo_reg_loss": float(np.mean(train_geo_reg_losses)),
            "pullback_l2": float(np.mean(train_pullback_l2)) if train_pullback_l2 else 0.0,
            "pullback_loss": float(np.mean(train_pullback_losses)) if train_pullback_losses else 0.0,
            "wsi_geo_reg_loss": float(np.mean(train_wsi_geo_reg_losses)),
            "rna_geo_reg_loss": float(np.mean(train_rna_geo_reg_losses)),
            "wsi_geo_step_reg_loss": float(np.mean(train_wsi_geo_step_reg_losses)),
            "rna_geo_step_reg_loss": float(np.mean(train_rna_geo_step_reg_losses)),
            "epoch_window_weight": float(np.mean(epoch_window_weights)) if epoch_window_weights else float("nan"),
            "window_metric_value": float(window_metric_now),
            "window_weight_now": float(w_now),
            "window_lower_now": float(window_lower_now),
            "window_upper_now": float(window_upper_now),
            "window_k_now": float(window_k_now),
            "window_k_left_now": float(window_k_left_now),
            "window_k_right_now": float(window_k_right_now),
            "window_center_now": float(window_center_now),
            "window_width_now": float(window_width_now),
            "train_loss_eval": float(train_loss_eval),
            "train_c_index": float(train_c_index),
            "val_c_index": float(val_c_index),
            "val_loss": float(val_loss),
            "val_c_index_ema": float(val_c_index_ema),
            "selection_metric": str(args.selection_metric),
            "selection_score": float(select_score),
            "test_c_index": float(test_c_index),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(
            f"[stage{stage}] epoch={epoch} "
            f"loss={row['loss']:.4f} task_loss={row['task_loss']:.4f} "
            f"w_prev={row['epoch_window_weight']:.4f} w_now={row['window_weight_now']:.4f} "
            f"pull={row['pullback_loss']:.4f} "
            f"train_c={row['train_c_index']:.4f} "
            f"val_c={row['val_c_index']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_c_ema={row['val_c_index_ema']:.4f} test_c={row['test_c_index']:.4f}"
        )
        if epoch >= int(args.selection_min_epochs) and stale_epochs >= int(args.early_stop_patience):
            print(
                f"early_stop stage={stage} epoch={epoch} "
                f"selection_metric={str(args.selection_metric)} best_epoch={best_epoch} stale={stale_epochs}"
            )
            break

        if scheduler is not None:
            scheduler.step()

    history_cols = [
        "stage",
        "epoch",
        "loss",
        "task_loss",
        "geo_reg_loss",
        "pullback_l2",
        "pullback_loss",
        "wsi_geo_reg_loss",
        "rna_geo_reg_loss",
        "wsi_geo_step_reg_loss",
        "rna_geo_step_reg_loss",
        "epoch_window_weight",
        "window_metric_value",
        "window_weight_now",
        "window_lower_now",
        "window_upper_now",
        "window_k_now",
        "window_k_left_now",
        "window_k_right_now",
        "window_center_now",
        "window_width_now",
        "train_loss_eval",
        "train_c_index",
        "val_c_index",
        "val_loss",
        "val_c_index_ema",
        "selection_metric",
        "selection_score",
        "test_c_index",
        "lr",
    ]
    write_csv(out_dir / "histories" / f"seed{int(args.seed)}.csv", history_rows, history_cols)

    window_loss_cfg = None
    if stage == 2:
        window_loss_cfg = {
            "metric": str(args.loss_window_metric),
            "policy": str(args.loss_window_policy),
            "lower": float(args.loss_window_lower),
            "upper": float(args.loss_window_upper),
            "k": float(args.loss_window_k),
            "A": float(args.loss_window_A),
            "A_effective": 1.0 if str(args.loss_window_policy) in ("onek", "twok") else float(args.loss_window_A),
            "eps": float(args.loss_window_eps),
            "center_mode": str(args.loss_window_center_mode),
            "center_quantile": float(args.loss_window_center_quantile),
            "best_mix_alpha": float(args.loss_window_best_mix_alpha),
            "width_mode": str(args.loss_window_width_mode),
            "width_mult": float(args.loss_window_width_mult),
            "width_quantile": float(args.loss_window_width_quantile),
            "width_min": float(args.loss_window_width_min),
            "width_max": float(args.loss_window_width_max),
            "transition": float(args.loss_window_transition),
            "K": float(args.loss_window_K),
            "K_left": None if args.loss_window_K_left is None else float(args.loss_window_K_left),
            "K_right": None if args.loss_window_K_right is None else float(args.loss_window_K_right),
        }
    final_test_c_index = float(history_rows[-1]["test_c_index"]) if history_rows else float("nan")
    save_json(
        out_dir / "summary.json",
        {
            "stage": stage,
            "mode": "censored_stage_survival_v2",
            "no_geo_contract": "enforced (hard guardrails, any geo / custom raises)",
            "model_variant": model_variant,
            "rna_mode": str(args.rna_mode),
            "rna_gene_sets_csv": str(gene_sets_csv) if gene_sets_csv is not None else None,
            "gene_annotation_gtf": str(gtf_path) if gtf_path is not None else None,
            "pool_method": str(args.pool_method),
            "gate_enabled": str(args.gate_enabled),
            "gate_enabled_resolved": bool(model.use_gate),
            "cross_modal_fusion": str(args.cross_modal_fusion),
            "wsi_geo_type": str(args.wsi_geo_type),
            "rna_geo_type": str(args.rna_geo_type),
            "geo_swap": bool(args.geo_swap),
            "use_multi_slide": bool(args.use_multi_slide),
            "multi_slide_mode": str(args.multi_slide_mode),
            "multi_slide_tile_budget_mode": str(args.multi_slide_tile_budget_mode),
            "wsi_feature_source": primary_source,
            "wsi_feature_source_aliases": sorted(accepted_sources - {primary_source}),
            "split_dir": str(split_dir),
            "target_col": str(args.target_col),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_min": float(args.lr_min),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "max_tiles": int(args.max_tiles),
            "n_train_cases": len(train_loader.dataset),
            "n_val_cases": len(val_loader.dataset),
            "n_test_cases": len(test_loader.dataset),
            "val_split_mode": str(args.val_split_mode),
            "val_time_bins": int(args.val_time_bins),
            "selection_metric": str(args.selection_metric),
            "val_ema_decay": float(args.val_ema_decay),
            "selection_min_epochs": int(args.selection_min_epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "pretrain": pretrain_summary,
            "window_loss_config": window_loss_cfg,
            "best_epoch": int(best_epoch),
            "best_val_c_index": best_val,
            "best_val_loss": best_val_loss,
            "best_select_score": best_select_score,
            "best_test_c_index": best_test,
            "best_train_c_index": best_train,
            "final_test_c_index": final_test_c_index,
        },
    )
    print(f"[DONE stage={stage}] summary.json: {out_dir / 'summary.json'}")
    if stage == 2:
        print(f"[DONE stage=2] final.pt: {out_dir / 'final.pt'}  (final_test_c_index={final_test_c_index:.4f})")


if __name__ == "__main__":
    main()
