from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data.paths import resolve_data_paths
from r2wsp.rna.gene_sets import load_gene_sets_csv
from r2wsp.rna.tokenizer import build_omics_spec
from r2wsp.data.tcga_dataset import _CohortTPMTsv


def _strip_gene_version(gene_id: str) -> str:
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


def _load_gene_id_to_symbol_map(gtf_path: Path) -> dict[str, str]:
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
            mapping[_strip_gene_version(gene_id)] = str(gene_name)
    if not mapping:
        raise ValueError(f"no gene_id -> gene_name mapping found in GTF: {gtf_path}")
    return mapping


def _neg_partial_log_likelihood(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    r = risk.index_select(0, order)
    e = event.index_select(0, order)
    log_cumsum = torch.logcumsumexp(r, dim=0)
    per = (log_cumsum - r) * e
    denom = e.sum().clamp_min(1.0)
    return per.sum() / denom


def _resolve_tpm_tsv(raw_rna_root: Path, cohort: str) -> Path:
    candidate = (raw_rna_root / "tpm_tsv" / f"{str(cohort).lower()}_tpm.tsv").resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"TPM TSV not found: {candidate}")
    return candidate


def _load_split(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_p = (split_dir / "train.csv").resolve()
    test_p = (split_dir / "test.csv").resolve()
    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"missing split csv under: {split_dir}")
    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)
    return train_df, test_df


def _build_pathway_means(
    tpm: _CohortTPMTsv,
    *,
    gene_sets_csv: Path,
    gene_id_to_symbol: dict[str, str] | None,
    case_ids: list[str],
) -> tuple[torch.Tensor, list[str]]:
    gene_sets = load_gene_sets_csv(gene_sets_csv)
    gene_names = [
        gene_id_to_symbol.get(_strip_gene_version(g), _strip_gene_version(g)) if gene_id_to_symbol is not None else _strip_gene_version(g)
        for g in tpm.genes
    ]
    spec = build_omics_spec(gene_names, gene_sets)
    idx_list = [idx.to(dtype=torch.long) for idx in spec.gene_indices]
    feats = torch.zeros((len(case_ids), len(idx_list)), dtype=torch.float32)
    for i, case_id in enumerate(case_ids):
        vec = tpm.get_case_vec(case_id)
        for j, idx in enumerate(idx_list):
            x = vec.index_select(0, idx)
            feats[i, j] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).mean()
    return feats, list(spec.pathway_names)


def fit_cox_teacher(
    X_train: torch.Tensor,
    time_train: torch.Tensor,
    event_train: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    w = torch.zeros((X_train.shape[1],), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w], lr=float(lr), weight_decay=float(weight_decay))
    best = math.inf
    best_w = None
    patience = 50
    bad = 0
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        risk = X_train.matmul(w)
        loss = _neg_partial_log_likelihood(risk, time_train, event_train)
        loss.backward()
        opt.step()
        v = float(loss.detach().cpu())
        if v + 1e-8 < best:
            best = v
            best_w = w.detach().clone()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_w is None:
        best_w = w.detach().clone()
    return best_w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, help="phase2_independent5/fold_k/eval_with_censored")
    p.add_argument("--cohort", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    args = p.parse_args()

    split_dir = Path(args.split_dir).resolve()
    cohort = str(args.cohort).upper()
    target_col = str(args.target_col)
    censor_col = f"{target_col.split('_', 1)[0]}_censorship"

    paths = resolve_data_paths()
    tpm_path = _resolve_tpm_tsv(paths.raw_rna_root, cohort)
    tpm = _CohortTPMTsv(tpm_path)

    gene_sets_csv = Path(args.rna_gene_sets_csv).resolve() if args.rna_gene_sets_csv is not None else (paths.raw_rna_root / "metadata" / "hallmarks_signatures.csv").resolve()
    if not gene_sets_csv.exists():
        raise FileNotFoundError(f"rna_gene_sets_csv not found: {gene_sets_csv}")

    gtf_path = Path(args.gene_annotation_gtf).resolve() if args.gene_annotation_gtf is not None else None
    gene_id_to_symbol = _load_gene_id_to_symbol_map(gtf_path) if gtf_path is not None else None

    train_df, test_df = _load_split(split_dir)
    for df in (train_df, test_df):
        if "case_id" not in df.columns:
            raise ValueError("split csv must contain case_id column")
        if target_col not in df.columns or censor_col not in df.columns:
            raise ValueError(f"split csv must contain {target_col} and {censor_col}")

    train_df = train_df[["case_id", target_col, censor_col]].copy()
    test_df = test_df[["case_id", target_col, censor_col]].copy()
    train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    test_df[target_col] = pd.to_numeric(test_df[target_col], errors="coerce")
    train_df[censor_col] = pd.to_numeric(train_df[censor_col], errors="coerce")
    test_df[censor_col] = pd.to_numeric(test_df[censor_col], errors="coerce")

    train_df = train_df.dropna(subset=[target_col, censor_col])
    test_df = test_df.dropna(subset=[target_col, censor_col])
    train_df = train_df[(train_df[target_col] >= 0) & (train_df[censor_col].isin([0, 1]))]
    test_df = test_df[(test_df[target_col] >= 0) & (test_df[censor_col].isin([0, 1]))]

    train_cases = train_df["case_id"].astype(str).tolist()
    all_cases = pd.concat([train_df["case_id"], test_df["case_id"]], ignore_index=True).astype(str).tolist()
    all_cases = sorted(set(all_cases))

    X_train, pathway_names = _build_pathway_means(
        tpm,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        case_ids=train_cases,
    )
    times = torch.tensor(train_df[target_col].to_numpy(dtype=np.float32, copy=False))
    censorship = torch.tensor(train_df[censor_col].to_numpy(dtype=np.float32, copy=False))
    events = 1.0 - censorship

    w = fit_cox_teacher(
        X_train,
        times,
        events,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        seed=int(args.seed),
    )

    X_all, _ = _build_pathway_means(
        tpm,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        case_ids=all_cases,
    )
    risk_all = X_all.matmul(w).detach().cpu().numpy().astype(np.float32)

    train_mask = np.array([c in set(train_cases) for c in all_cases], dtype=bool)
    mu = float(risk_all[train_mask].mean()) if train_mask.any() else float(risk_all.mean())
    sigma = float(risk_all[train_mask].std(ddof=0)) if train_mask.any() else float(risk_all.std(ddof=0))
    sigma = max(sigma, 1e-6)
    z = (risk_all - mu) / sigma

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"case_id": all_cases, "pseudo_risk": z})
    out_df.to_csv(out_csv, index=False)

    meta = {
        "cohort": cohort,
        "split_dir": str(split_dir),
        "target_col": target_col,
        "censor_col": censor_col,
        "tpm_tsv": str(tpm_path),
        "gene_sets_csv": str(gene_sets_csv),
        "gtf_path": str(gtf_path) if gtf_path is not None else None,
        "n_train_cases": int(len(train_cases)),
        "n_all_cases": int(len(all_cases)),
        "pathway_count": int(len(pathway_names)),
        "pathway_names": pathway_names,
        "teacher": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "risk_mean_train": mu,
            "risk_std_train": sigma,
            "w": w.detach().cpu().numpy().astype(np.float32).tolist(),
        },
    }
    meta_path = out_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
