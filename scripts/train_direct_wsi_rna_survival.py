from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import re

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
    m = re.search(r"TCGA_([A-Z0-9]+)_OVERALL_SURVIVAL", str(split_dir).upper())
    return m.group(1) if m is not None else None


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
    max_tiles: int,
    rna_mode: str,
    gene_sets_csv: str | Path | None,
    gene_id_to_symbol: dict[str, str] | None,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    use_multi_slide: bool,
    multi_slide_tile_budget_mode: str,
) -> DataLoader:
    ds = TCGAMultimodalDataset(
        rows,
        split=None,
        seed=seed,
        max_tiles=max_tiles,
        rna_mode=rna_mode,
        rna_gene_sets_csv=gene_sets_csv,
        rna_gene_id_to_symbol=gene_id_to_symbol,
        use_multi_slide=use_multi_slide,
        multi_slide_tile_budget_mode=multi_slide_tile_budget_mode,
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


def main() -> None:
    p = argparse.ArgumentParser(description="Direct-input WSI+RNA survival baseline without consuming MMP latent exports.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--wsi_feature_source", default="plip_luad_256")
    p.add_argument("--split_dir", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--max_tiles", type=int, default=256)
    p.add_argument("--rna_mode", default="vec", choices=["vec", "omics", "omics_attn"])
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
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
    p.add_argument("--fusion_mode", default="concat", choices=["concat", "gated"], help="Deprecated alias; gated maps to model_variant=gate_only when model_variant=baseline.")
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
    p.add_argument("--out_dir", default="outputs/direct_wsi_rna_survival")
    args = p.parse_args()

    set_seed(int(args.seed))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)

    model_variant = str(args.model_variant)
    if model_variant == "baseline" and str(args.fusion_mode) == "gated":
        model_variant = "gate_only"
    gate_enabled = None
    if str(args.gate_enabled) == "true":
        gate_enabled = True
    elif str(args.gate_enabled) == "false":
        gate_enabled = False

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
    cohort_hint = infer_cohort_from_split_dir(split_dir)
    rows = apply_cohort_hint(
        all_rows if bool(args.use_multi_slide) else dedupe_rows_by_case(all_rows, source_name=str(args.wsi_feature_source)),
        cohort_hint=cohort_hint,
        reference_rows=all_rows,
    )
    rows = [r for r in rows if str(r.wsi_feature_source) == str(args.wsi_feature_source)]
    case_table, train_case_ids, test_case_ids = load_official_split(split_dir, str(args.target_col))
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

    train_loader = build_loader(
        train_rows,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        max_tiles=int(args.max_tiles),
        rna_mode=dataset_rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
    )
    train_eval_loader = build_loader(
        train_rows,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        max_tiles=int(args.max_tiles),
        rna_mode=dataset_rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
    )
    val_loader = build_loader(
        val_rows,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        max_tiles=int(args.max_tiles),
        rna_mode=dataset_rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
    )
    test_loader = build_loader(
        test_rows,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        max_tiles=int(args.max_tiles),
        rna_mode=dataset_rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
    )

    sample_batch = next(iter(train_loader))
    tile_dim = int(sample_batch.tile_tokens.shape[-1])
    rna_dim = int(sample_batch.rna_vec.shape[-1]) if sample_batch.rna_vec is not None else 1
    rna_omic_sizes = [int(x.shape[-1]) for x in sample_batch.rna_omics] if sample_batch.rna_omics is not None else None

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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_val = float("-inf")
    best_test = float("nan")
    best_train = float("nan")
    best_val_loss = float("inf")
    best_select_score = float("-inf") if str(args.selection_metric) != "val_loss" else float("inf")
    val_c_index_ema: float | None = None
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
        for batch in train_loader:
            times, events = gather_labels(batch.case_id, case_table, device)
            risk, aux = model(
                tile_tokens=batch.tile_tokens.to(device),
                tile_xy=batch.tile_xy.to(device),
                tile_attn_mask=batch.tile_attn_mask.to(device),
                slide_ids=batch.slide_ids.to(device),
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
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
            loss = task_loss + geo_reg_loss
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
                },
                out_dir / "best.pt",
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "loss": float(np.mean(train_losses)),
            "task_loss": float(np.mean(train_task_losses)),
            "geo_reg_loss": float(np.mean(train_geo_reg_losses)),
            "wsi_geo_reg_loss": float(np.mean(train_wsi_geo_reg_losses)),
            "rna_geo_reg_loss": float(np.mean(train_rna_geo_reg_losses)),
            "wsi_geo_step_reg_loss": float(np.mean(train_wsi_geo_step_reg_losses)),
            "rna_geo_step_reg_loss": float(np.mean(train_rna_geo_step_reg_losses)),
            "train_loss_eval": float(train_loss_eval),
            "train_c_index": float(train_c_index),
            "val_c_index": float(val_c_index),
            "val_loss": float(val_loss),
            "val_c_index_ema": float(val_c_index_ema),
            "selection_metric": str(args.selection_metric),
            "selection_score": float(select_score),
            "test_c_index": float(test_c_index),
        }
        history_rows.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.4f} task_loss={row['task_loss']:.4f} "
            f"geo_reg_loss={row['geo_reg_loss']:.4f} train_loss_eval={row['train_loss_eval']:.4f} "
            f"train_c_index={row['train_c_index']:.4f} "
            f"val_c_index={row['val_c_index']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_c_index_ema={row['val_c_index_ema']:.4f} test_c_index={row['test_c_index']:.4f}"
        )
        if epoch >= int(args.selection_min_epochs) and stale_epochs >= int(args.early_stop_patience):
            print(
                f"early_stop epoch={epoch} selection_metric={str(args.selection_metric)} "
                f"best_epoch={best_epoch} stale_epochs={stale_epochs}"
            )
            break

    write_csv(
        out_dir / "histories" / f"seed{int(args.seed)}.csv",
        history_rows,
        [
            "epoch",
            "loss",
            "task_loss",
            "geo_reg_loss",
            "wsi_geo_reg_loss",
            "rna_geo_reg_loss",
            "wsi_geo_step_reg_loss",
            "rna_geo_step_reg_loss",
            "train_loss_eval",
            "train_c_index",
            "val_c_index",
            "val_loss",
            "val_c_index_ema",
            "selection_metric",
            "selection_score",
            "test_c_index",
        ],
    )
    save_json(
        out_dir / "summary.json",
        {
            "mode": "direct_input_survival_baseline",
            "model_variant": model_variant,
            "rna_mode": str(args.rna_mode),
            "rna_gene_sets_csv": str(gene_sets_csv) if gene_sets_csv is not None else None,
            "gene_annotation_gtf": str(gtf_path) if gtf_path is not None else None,
            "pool_method": str(args.pool_method),
            "gate_enabled": str(args.gate_enabled),
            "gate_enabled_resolved": bool(model.use_gate),
            "cross_modal_fusion": str(args.cross_modal_fusion),
            "wsi_geo_type": str(args.wsi_geo_type),
            "wsi_geo_num_points": int(args.wsi_geo_num_points),
            "wsi_geo_coord_dim": int(args.wsi_geo_coord_dim),
            "wsi_geo_output": str(args.wsi_geo_output),
            "wsi_geo_reg_type": str(args.wsi_geo_reg_type),
            "wsi_geo_reg_lambda": float(args.wsi_geo_reg_lambda),
              "wsi_geo_step_target": float(args.wsi_geo_step_target),
              "wsi_geo_step_target_lambda": float(args.wsi_geo_step_target_lambda),
            "wsi_geo_position": str(args.wsi_geo_position),
            "wsi_geo_fusion": str(args.wsi_geo_fusion),
            "rna_geo_type": str(args.rna_geo_type),
            "rna_geo_num_points": int(args.rna_geo_num_points),
            "rna_geo_coord_dim": int(args.rna_geo_coord_dim),
            "rna_geo_output": str(args.rna_geo_output),
            "rna_geo_reg_type": str(args.rna_geo_reg_type),
            "rna_geo_reg_lambda": float(args.rna_geo_reg_lambda),
              "rna_geo_step_target": float(args.rna_geo_step_target),
              "rna_geo_step_target_lambda": float(args.rna_geo_step_target_lambda),
            "rna_geo_position": str(args.rna_geo_position),
            "rna_geo_fusion": str(args.rna_geo_fusion),
            "geo_swap": bool(args.geo_swap),
            "use_multi_slide": bool(args.use_multi_slide),
            "multi_slide_mode": str(args.multi_slide_mode),
            "multi_slide_tile_budget_mode": str(args.multi_slide_tile_budget_mode),
            "wsi_feature_source": str(args.wsi_feature_source),
            "split_dir": str(split_dir),
            "target_col": str(args.target_col),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "n_train_cases": len(train_loader.dataset),
            "n_val_cases": len(val_loader.dataset),
            "n_test_cases": len(test_loader.dataset),
            "val_split_mode": str(args.val_split_mode),
            "val_time_bins": int(args.val_time_bins),
            "selection_metric": str(args.selection_metric),
            "val_ema_decay": float(args.val_ema_decay),
            "selection_min_epochs": int(args.selection_min_epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "best_epoch": best_epoch,
            "best_val_c_index": best_val,
            "best_val_loss": best_val_loss,
            "best_select_score": best_select_score,
            "best_test_c_index": best_test,
            "best_train_c_index": best_train,
            "final_test_c_index": history_rows[-1]["test_c_index"] if history_rows else float("nan"),
        },
    )


if __name__ == "__main__":
    main()
