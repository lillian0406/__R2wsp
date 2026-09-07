from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _infer_feature_tag(feature_root: Path) -> str:
    if feature_root.name in {"feats_h5", "feats_pt"} and feature_root.parent.name:
        return feature_root.parent.name
    return feature_root.name


def _default_out_root(feature_root: Path, cohort: str, feature_tag: str) -> Path:
    for parent in [feature_root, *feature_root.parents]:
        if parent.name == "data" and parent.parent.exists():
            return (parent / "bridges" / f"mmp_{cohort.lower()}_{feature_tag}_official_dss").resolve()
    return (feature_root.parent / f"mmp_{cohort.lower()}_{feature_tag}_official_dss_bridge").resolve()


def _infer_feature_suffix(feature_root: Path, user_value: str | None) -> str:
    if user_value:
        suffix = str(user_value)
        return suffix if suffix.startswith(".") else f".{suffix}"
    if feature_root.name == "feats_h5":
        return ".h5"
    if feature_root.name == "feats_pt":
        return ".pt"
    counts = {
        ".h5": sum(1 for _ in feature_root.rglob("*.h5")),
        ".pt": sum(1 for _ in feature_root.rglob("*.pt")),
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise FileNotFoundError(f"no .h5 or .pt features found under {feature_root}")
    return best


def _parse_mag_patch(feature_root: Path) -> tuple[int | None, int | None]:
    pattern = re.compile(r"extracted_mag(\d+)x_patch(\d+)_fp", re.IGNORECASE)
    for parent in [feature_root, *feature_root.parents]:
        match = pattern.search(str(parent))
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def _index_feature_files(feature_root: Path, feature_suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(feature_root.rglob(f"*{feature_suffix}")):
        if path.is_file():
            out[path.stem.lower()] = path
    return out


def _filter_split_df(df: pd.DataFrame, feature_index: dict[str, Path]) -> pd.DataFrame:
    out = df.copy()
    out["slide_id"] = out["slide_id"].astype(str)
    keep = out["slide_id"].str.lower().isin(feature_index.keys())
    return out.loc[keep].reset_index(drop=True)


def _censorship_counts(df: pd.DataFrame, censorship_col: str) -> dict[str, int] | None:
    if censorship_col not in df.columns:
        return None
    counts = df[censorship_col].astype(int).value_counts().to_dict()
    return {str(int(k)): int(v) for k, v in counts.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare official MMP survival splits against an existing MMP-style WSI feature directory.")
    p.add_argument("--mmp-root", required=True)
    p.add_argument("--feature-root", required=True, help="Directory containing slide-level feature files, typically feats_h5 or feats_pt.")
    p.add_argument("--feature-suffix", default=None, help="Feature file suffix, for example .h5 or .pt. Auto-inferred when omitted.")
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--target-col", default="dss_survival_days")
    p.add_argument("--out-root", default=None, help="Bridge root for filtered split CSVs and summary metadata.")
    args = p.parse_args()

    mmp_root = Path(args.mmp_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    if not feature_root.exists():
        raise FileNotFoundError(feature_root)

    cohort = str(args.cohort).upper()
    feature_suffix = _infer_feature_suffix(feature_root, args.feature_suffix)
    feature_tag = _infer_feature_tag(feature_root)
    patch_mag, patch_size = _parse_mag_patch(feature_root)
    split_root = (mmp_root / "src" / "splits" / "survival").resolve()
    feature_index = _index_feature_files(feature_root, feature_suffix)
    if not feature_index:
        raise FileNotFoundError(f"no feature files with suffix {feature_suffix} found under {feature_root}")

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else _default_out_root(feature_root, cohort, feature_tag)
    )
    splits_out_root = out_root / "splits" / "survival"
    summary_path = out_root / "summary.json"

    overall = {
        "cohort": cohort,
        "target_col": str(args.target_col),
        "mmp_root": str(mmp_root),
        "feature_root": str(feature_root),
        "feature_suffix": feature_suffix,
        "feature_tag": feature_tag,
        "patch_mag": patch_mag,
        "patch_size": patch_size,
        "out_root": str(out_root),
        "splits_out_root": str(splits_out_root),
        "feature_files": int(len(feature_index)),
        "folds": [],
    }

    censorship_col = f"{str(args.target_col).split('_')[0]}_censorship"

    for k in range(int(args.n_folds)):
        in_dir = split_root / f"TCGA_{cohort}_overall_survival_k={k}"
        train_in = in_dir / "train.csv"
        test_in = in_dir / "test.csv"
        if not train_in.exists() or not test_in.exists():
            raise FileNotFoundError(f"missing official split files: {train_in} / {test_in}")

        train_full = pd.read_csv(train_in)
        test_full = pd.read_csv(test_in)
        train_df = _filter_split_df(train_full, feature_index)
        test_df = _filter_split_df(test_full, feature_index)

        out_dir = splits_out_root / f"TCGA_{cohort}_overall_survival_k={k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

        overall["folds"].append(
            {
                "k": int(k),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_cases": int(train_df["case_id"].astype(str).nunique()) if "case_id" in train_df.columns else None,
                "test_cases": int(test_df["case_id"].astype(str).nunique()) if "case_id" in test_df.columns else None,
                "train_censorship_counts": _censorship_counts(train_df, censorship_col),
                "test_censorship_counts": _censorship_counts(test_df, censorship_col),
                "missing_train": int((~train_full["slide_id"].astype(str).str.lower().isin(feature_index.keys())).sum()),
                "missing_test": int((~test_full["slide_id"].astype(str).str.lower().isin(feature_index.keys())).sum()),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("\nSuggested runner command:")
    print(
        "python scripts/run_mmp_official_luad_plip_5fold.py "
        f"--mmp-root {mmp_root} "
        f"--bridge-root {out_root} "
        f"--data-source {feature_root} "
        f"--feature-dir-name {feature_root.name} "
        f"--feature-name {feature_tag} "
        f"--cohort {cohort}"
    )


if __name__ == "__main__":
    main()
