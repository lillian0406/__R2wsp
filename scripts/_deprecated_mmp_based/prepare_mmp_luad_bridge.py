from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets

TCGA_CASE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def _stable_split(case_id: str, *, seed: int, val_frac: float, test_frac: float) -> str:
    key = f"split|{case_id}|{seed}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    r = int(h[:8], 16) / float(16**8)
    if r < float(test_frac):
        return "test"
    if r < float(test_frac) + float(val_frac):
        return "val"
    return "train"


def _fold_split(case_id: str, *, seed: int, k: int, n_folds: int, val_frac: float) -> str:
    if n_folds <= 1:
        raise ValueError("n_folds must be > 1 for fold split")
    key_test = f"foldtest|{case_id}|{seed}".encode("utf-8")
    h_test = int(hashlib.sha256(key_test).hexdigest()[:8], 16)
    is_test = (h_test % int(n_folds)) == int(k)
    if is_test:
        return "test"
    train_pool = 1.0 - (1.0 / float(n_folds))
    if train_pool <= 0:
        return "train"
    val_in_train = float(val_frac) / train_pool
    val_in_train = max(0.0, min(0.95, val_in_train))
    key_val = f"foldval|{case_id}|{seed}|{k}".encode("utf-8")
    r = int(hashlib.sha256(key_val).hexdigest()[:8], 16) / float(16**8)
    if r < val_in_train:
        return "val"
    return "train"


def _compute_os_fields(row: pd.Series) -> tuple[float | None, int | None]:
    raw_vital = row.get("vital_status", "")
    vital = str(raw_vital).strip().lower()
    days_to_death = pd.to_numeric(row.get("days_to_death"), errors="coerce")
    days_to_follow = pd.to_numeric(row.get("days_to_last_follow_up"), errors="coerce")
    is_dead = vital in {"dead", "deceased", "0", "false"}
    is_alive = vital in {"alive", "living", "1", "true"}
    if not is_dead and not is_alive:
        if str(raw_vital).strip() in {"0", "1"}:
            is_dead = str(raw_vital).strip() == "0"
            is_alive = str(raw_vital).strip() == "1"

    if is_dead:
        if not pd.isna(days_to_death):
            return float(days_to_death), 0
        if not pd.isna(days_to_follow):
            return float(days_to_follow), 0
        return None, None

    if is_alive:
        if not pd.isna(days_to_follow):
            return float(days_to_follow), 1
        if not pd.isna(days_to_death):
            return float(days_to_death), 1
        return None, None
    return None, None


def _select_case_ids(case_ids: list[str], *, seed: int, max_cases: int | None) -> list[str]:
    uniq = sorted(set(case_ids))
    if max_cases is None or int(max_cases) <= 0 or len(uniq) <= int(max_cases):
        return uniq
    ranked = sorted(
        uniq,
        key=lambda x: hashlib.sha256(f"sample|{x}|{int(seed)}".encode("utf-8")).hexdigest(),
    )
    return sorted(ranked[: int(max_cases)])


def _select_case_ids_balanced(
    case_df: pd.DataFrame, *, seed: int, max_cases: int | None, require_both: bool
) -> list[str]:
    if case_df.empty:
        return []
    if max_cases is None or int(max_cases) <= 0:
        return sorted(case_df["case_id"].astype(str).unique().tolist())

    dead = sorted(case_df[case_df["os_censorship"] == 0]["case_id"].astype(str).unique().tolist())
    alive = sorted(case_df[case_df["os_censorship"] == 1]["case_id"].astype(str).unique().tolist())
    if require_both and (not dead or not alive):
        raise ValueError(f"cannot satisfy censorship diversity: dead={len(dead)} alive={len(alive)}")

    def _rank(xs: list[str], tag: str) -> list[str]:
        return sorted(xs, key=lambda x: hashlib.sha256(f"sample|{tag}|{x}|{int(seed)}".encode("utf-8")).hexdigest())

    k = int(max_cases)
    if require_both and k >= 2:
        n_dead = min(len(dead), max(1, k // 2))
        n_alive = min(len(alive), k - n_dead)
        if n_alive == 0:
            n_alive = 1
            n_dead = min(len(dead), k - 1)
        chosen = _rank(dead, "dead")[:n_dead] + _rank(alive, "alive")[:n_alive]
        return sorted(chosen)

    chosen = _rank(dead, "dead") + _rank(alive, "alive")
    return sorted(chosen[:k])


def _normalize_case_id(text: object) -> str | None:
    s = str(text).upper()
    m = TCGA_CASE_RE.search(s)
    if m is None:
        return None
    return m.group(0)


def _load_rna_matrix(path: Path, *, case_ids: list[str], log1p: bool) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    if len(header) < 3:
        raise ValueError(f"invalid TPM TSV: {path}")
    gene_col = header[0]
    case_to_cols: dict[str, list[str]] = {}
    for col in header[1:]:
        if col is None or str(col).strip() == "":
            continue
        case_id = _normalize_case_id(col)
        if case_id is None:
            continue
        case_to_cols.setdefault(case_id, []).append(col)
    selected_feature_cols: list[str] = []
    for case_id in case_ids:
        selected_feature_cols.extend(case_to_cols.get(case_id, []))
    selected_cols = [gene_col, *selected_feature_cols]
    selected_cols = [str(c) for c in selected_cols if c is not None and str(c).strip() != ""]
    if len(selected_cols) < 2:
        raise ValueError(f"none of the requested cases are present in RNA TSV: {path}")
    df = pd.read_csv(path, sep="\t", usecols=selected_cols)
    df = df.set_index(gene_col)
    rename_map = {col: _normalize_case_id(col) for col in df.columns}
    df = df.rename(columns=rename_map)
    df = df.loc[:, [c for c in df.columns if c is not None]]
    if df.shape[1] == 0:
        raise ValueError(f"none of the requested cases are present in RNA TSV: {path}")
    if len(set(df.columns)) != len(df.columns):
        df = df.transpose().groupby(level=0).mean().transpose()
    keep = [c for c in case_ids if c in df.columns]
    if not keep:
        raise ValueError(f"none of the requested cases are present in RNA TSV after case-id normalization: {path}")
    mat = df[keep].transpose().copy()
    mat.index.name = "case_id"
    if log1p:
        mat = np.log1p(mat.astype(np.float32))
    return mat


def _export_feats_pt(rows: list[dict[str, str]], *, feats_dir: Path, overwrite: bool) -> int:
    feats_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        src = Path(row["token_path"])
        dst = feats_dir / f"{row['slide_id']}.pt"
        if dst.exists() and not overwrite:
            continue
        arr = np.load(src, allow_pickle=False)["feat"]
        feat = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        torch.save(feat, dst)
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare LUAD bridge assets so MMP can run on existing R2wsp data.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--token_source", default="plip_luad_256")
    p.add_argument("--feature_name", default="plip_luad_256")
    p.add_argument("--patch_mag", type=int, default=20)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--n_folds", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--require_both_censorship", action="store_true", default=False)
    p.add_argument("--split_mode", choices=["stable", "train_only"], default=None)
    p.add_argument("--log1p_rna", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--out_root", default=None)
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    cohort = str(args.cohort).upper()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root is not None
        else (paths.data_root / "bridges" / f"mmp_{cohort.lower()}_{args.token_source}").resolve()
    )

    inv = scan_all_assets(paths)
    rows = build_index(inv)
    rows = [r for r in rows if r.cohort == cohort and r.token_source == args.token_source and r.case_id and r.rna_path and r.clinical_path]
    if not rows:
        raise ValueError(f"no rows found for cohort={cohort} token_source={args.token_source}")

    clinical_path = Path(rows[0].clinical_path)
    rna_path = Path(rows[0].rna_path)
    clinical = pd.read_csv(clinical_path)
    clinical["case_id_bridge"] = clinical["submitter_id"].astype(str)

    os_values = clinical.apply(_compute_os_fields, axis=1, result_type="expand")
    clinical["os_survival_days"] = os_values[0]
    clinical["os_censorship"] = os_values[1]
    clinical = clinical.dropna(subset=["case_id_bridge", "os_survival_days", "os_censorship"])
    clinical["os_survival_days"] = clinical["os_survival_days"].astype(float)
    clinical["os_censorship"] = clinical["os_censorship"].astype(int)
    clinical = clinical.drop_duplicates(subset=["case_id_bridge"], keep="first").set_index("case_id_bridge")

    bridge_rows: list[dict[str, str]] = []
    for r in rows:
        case_id = str(r.case_id)
        if case_id not in clinical.index:
            continue
        bridge_rows.append(
            {
                "case_id": case_id,
                "slide_id": str(r.slide_id),
                "token_path": str(r.token_path),
                "os_survival_days": float(clinical.loc[case_id, "os_survival_days"]),
                "os_censorship": int(clinical.loc[case_id, "os_censorship"]),
                "vital_status": str(clinical.loc[case_id, "vital_status"]),
            }
        )
    if not bridge_rows:
        raise ValueError("no overlapping rows remain after clinical survival filtering")

    bridge_df = pd.DataFrame(bridge_rows).drop_duplicates(subset=["case_id", "slide_id"]).sort_values(["case_id", "slide_id"])
    case_df = bridge_df.drop_duplicates(subset=["case_id"])[["case_id", "os_censorship", "os_survival_days"]].copy()
    valid_case_ids = _select_case_ids_balanced(
        case_df,
        seed=int(args.seed),
        max_cases=int(args.max_cases) if int(args.max_cases) > 0 else None,
        require_both=bool(args.require_both_censorship),
    )
    bridge_df = bridge_df[bridge_df["case_id"].isin(valid_case_ids)].copy()

    rna_clean = _load_rna_matrix(rna_path, case_ids=valid_case_ids, log1p=bool(args.log1p_rna))
    valid_case_ids = [c for c in valid_case_ids if c in rna_clean.index]
    bridge_df = bridge_df[bridge_df["case_id"].isin(valid_case_ids)].copy()
    if bridge_df.empty:
        raise ValueError("no rows remain after intersecting with RNA cases")

    feature_root = out_root / "histology" / f"extracted_mag{int(args.patch_mag)}x_patch{int(args.patch_size)}_fp" / str(args.feature_name)
    feats_dir = feature_root / "feats_pt"
    omics_dir = out_root / "omics" / "hallmarks" / cohort

    written_pt = _export_feats_pt(bridge_df.to_dict("records"), feats_dir=feats_dir, overwrite=bool(args.overwrite))

    omics_dir.mkdir(parents=True, exist_ok=True)
    rna_clean = rna_clean.loc[sorted(set(bridge_df["case_id"].tolist()))]
    rna_clean.to_csv(omics_dir / "rna_clean.csv")

    split_mode = str(args.split_mode) if args.split_mode is not None else ("train_only" if int(args.max_cases) > 0 else "stable")
    all_cases = sorted(set(bridge_df["case_id"].tolist()))
    n_folds = int(args.n_folds)
    if n_folds < 1:
        raise ValueError("--n_folds must be >= 1")

    split_counts_by_fold: dict[str, dict[str, dict[str, int]]] = {}
    split_dirs: list[Path] = []

    fold_ks = [0] if n_folds == 1 else list(range(n_folds))
    for k in fold_ks:
        split_dir = out_root / "splits" / "survival" / f"TCGA_{cohort}_overall_survival_k={k}"
        split_dirs.append(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        split_to_cases: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        if split_mode == "train_only":
            split_to_cases["train"] = all_cases
        else:
            for case_id in all_cases:
                if n_folds > 1:
                    split = _fold_split(
                        case_id,
                        seed=int(args.seed),
                        k=int(k),
                        n_folds=int(n_folds),
                        val_frac=float(args.val_frac),
                    )
                else:
                    split = _stable_split(
                        case_id,
                        seed=int(args.seed),
                        val_frac=float(args.val_frac),
                        test_frac=float(args.test_frac),
                    )
                split_to_cases[split].append(case_id)

        split_counts: dict[str, dict[str, int]] = {}
        for split, case_ids in split_to_cases.items():
            part = bridge_df[bridge_df["case_id"].isin(case_ids)].copy()
            part.to_csv(split_dir / f"{split}.csv", index=False)
            cens_counts = (
                part.drop_duplicates(subset=["case_id"])["os_censorship"].astype(int).value_counts().to_dict()
                if not part.empty
                else {}
            )
            split_counts[split] = {
                "rows": int(len(part)),
                "cases": int(part["case_id"].nunique()),
                "censorship_counts": {str(kk): int(vv) for kk, vv in sorted(cens_counts.items())},
            }
        split_counts_by_fold[f"k={k}"] = split_counts

    overall_cens = (
        bridge_df.drop_duplicates(subset=["case_id"])["os_censorship"].astype(int).value_counts().to_dict()
    )

    summary = {
        "cohort": cohort,
        "token_source": args.token_source,
        "clinical_path": str(clinical_path),
        "rna_path": str(rna_path),
        "out_root": str(out_root),
        "feature_dir": str(feats_dir),
        "omics_dir": str(out_root / "omics"),
        "n_folds": int(n_folds),
        "split_dirs": [str(p) for p in split_dirs],
        "max_cases": int(args.max_cases),
        "split_mode": split_mode,
        "slides": int(len(bridge_df)),
        "cases": int(bridge_df["case_id"].nunique()),
        "censorship_counts": {str(k): int(v) for k, v in sorted(overall_cens.items())},
        "rna_cases": int(rna_clean.shape[0]),
        "genes": int(rna_clean.shape[1]),
        "written_pt": int(written_pt),
        "split_counts": split_counts_by_fold,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nSuggested MMP smoke run:")
    suggested_split_names = "train" if split_mode == "train_only" else "train,val,test"
    suggested_split_dir = split_dirs[0]
    print(
        "PYTHONPATH=/root/autodl-tmp/_refs/MMP-main/src "
        "python src/training/main_survival.py "
        f"--data_source {feats_dir} "
        f"--omics_dir {out_root / 'omics'} "
        f"--split_dir {suggested_split_dir} "
        f"--split_names {suggested_split_names} "
        f"--task {cohort}_survival "
        "--target_col os_survival_days "
        "--model_histo_type MIL "
        "--model_histo_config MIL_default "
        "--model_mm_type survpath "
        "--in_dim 512 "
        "--bag_size 256 "
        "--batch_size 1 "
        "--max_epochs 1 "
        "--num_workers 0 "
        f"--results_dir {out_root / 'results'}"
    )


if __name__ == "__main__":
    main()
