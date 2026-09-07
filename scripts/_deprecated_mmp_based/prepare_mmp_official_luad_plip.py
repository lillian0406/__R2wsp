from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import resolve_data_paths


def _build_npz_index(token_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in token_dir.rglob("*.npz"):
        index[p.stem.lower()] = p
    return index


def _export_feats_pt(split_df: pd.DataFrame, *, npz_index: dict[str, Path], feats_dir: Path, overwrite: bool) -> int:
    feats_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for slide_id in split_df["slide_id"].astype(str).tolist():
        src = npz_index.get(slide_id.lower())
        if src is None:
            continue
        dst = feats_dir / f"{slide_id}.pt"
        if dst.exists() and not overwrite:
            continue
        arr = np.load(src, allow_pickle=False)["feat"]
        feat = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        torch.save(feat, dst)
        written += 1
    return written


def _filter_split_df(df: pd.DataFrame, *, npz_index: dict[str, Path]) -> pd.DataFrame:
    df = df.copy()
    df["slide_id"] = df["slide_id"].astype(str)
    keep = df["slide_id"].str.lower().isin(npz_index.keys())
    return df.loc[keep].reset_index(drop=True)


def _censorship_counts(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in df["dss_censorship"].astype(int).value_counts().to_dict().items():
        out[str(int(k))] = int(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare PLIP tokens for running MMP on official LUAD DSS splits (subset of available slides).")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--token_source", default="plip_luad_256")
    p.add_argument("--feature_name", default="plip_luad_256")
    p.add_argument("--patch_mag", type=int, default=20)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--mmp_root", required=True)
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--out_root", default=None)
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    cohort = str(args.cohort).upper()
    mmp_root = Path(args.mmp_root).resolve()
    split_root = mmp_root / "src" / "splits" / "survival"

    token_dir = (paths.data_root / "tokens" / args.token_source).resolve()
    npz_index = _build_npz_index(token_dir)

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root is not None
        else (paths.data_root / "bridges" / f"mmp_{cohort.lower()}_{args.token_source}_official_dss").resolve()
    )

    histo_root = out_root / "histology" / f"extracted_mag{int(args.patch_mag)}x_patch{int(args.patch_size)}_fp" / args.feature_name / "feats_pt"
    splits_out_root = out_root / "splits" / "survival"

    overall = {
        "cohort": cohort,
        "token_source": args.token_source,
        "token_dir": str(token_dir),
        "npz_files": int(len(npz_index)),
        "mmp_root": str(mmp_root),
        "out_root": str(out_root),
        "feature_dir": str(histo_root),
        "folds": [],
    }

    total_written = 0
    for k in range(int(args.n_folds)):
        in_dir = split_root / f"TCGA_{cohort}_overall_survival_k={k}"
        train_in = in_dir / "train.csv"
        test_in = in_dir / "test.csv"
        if not train_in.exists() or not test_in.exists():
            raise FileNotFoundError(f"missing official split files: {train_in} / {test_in}")

        train_df = pd.read_csv(train_in)
        test_df = pd.read_csv(test_in)
        train_df = _filter_split_df(train_df, npz_index=npz_index)
        test_df = _filter_split_df(test_df, npz_index=npz_index)

        out_dir = splits_out_root / f"TCGA_{cohort}_overall_survival_k={k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

        written_k = _export_feats_pt(pd.concat([train_df, test_df], ignore_index=True), npz_index=npz_index, feats_dir=histo_root, overwrite=bool(args.overwrite))
        total_written += written_k

        fold_summary = {
            "k": int(k),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_cases": int(train_df["case_id"].astype(str).nunique()) if "case_id" in train_df.columns else None,
            "test_cases": int(test_df["case_id"].astype(str).nunique()) if "case_id" in test_df.columns else None,
            "train_censorship_counts": _censorship_counts(train_df) if "dss_censorship" in train_df.columns else None,
            "test_censorship_counts": _censorship_counts(test_df) if "dss_censorship" in test_df.columns else None,
            "missing_train": int((pd.read_csv(train_in)["slide_id"].astype(str).str.lower().isin(npz_index.keys()) == False).sum()),
            "missing_test": int((pd.read_csv(test_in)["slide_id"].astype(str).str.lower().isin(npz_index.keys()) == False).sum()),
        }
        overall["folds"].append(fold_summary)

    overall["written_pt"] = int(total_written)
    print(json.dumps(overall, indent=2, ensure_ascii=False))

    suggested = (
        "PYTHONPATH={mmp_src} python training/main_survival.py "
        "--data_source {data_source} "
        "--omics_dir {omics_dir} "
        "--split_dir {split_dir} "
        "--split_names train,test "
        "--task {task} "
        "--target_col dss_survival_days "
        "--model_histo_type MIL --model_histo_config MIL_default --model_mm_type survpath "
        "--in_dim 512 --bag_size 256 --batch_size 1 --max_epochs 50 --num_workers 4 "
        "--results_dir {results_dir}"
    ).format(
        mmp_src=str((mmp_root / "src").resolve()),
        data_source=str(histo_root),
        omics_dir=str((mmp_root / "src" / "data_csvs" / "rna").resolve()),
        split_dir=str((splits_out_root / f"TCGA_{cohort}_overall_survival_k=0").resolve()),
        task=f"{cohort}_survival",
        results_dir=str((out_root / "results").resolve()),
    )
    print("\nSuggested MMP run (k=0):")
    print(suggested)


if __name__ == "__main__":
    main()

