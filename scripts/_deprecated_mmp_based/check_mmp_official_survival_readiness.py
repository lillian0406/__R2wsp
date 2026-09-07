from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _collect_needed_slides(split_root: Path, cohort: str, n_folds: int) -> list[str]:
    needed: set[str] = set()
    for k in range(int(n_folds)):
        fold_dir = split_root / f"TCGA_{cohort}_overall_survival_k={k}"
        for split_name in ("train.csv", "test.csv"):
            split_path = fold_dir / split_name
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            df = pd.read_csv(split_path)
            needed.update(df["slide_id"].astype(str).str.lower().tolist())
    return sorted(needed)


def _index_stems(root: Path, suffix: str) -> set[str]:
    if not root.exists():
        return set()
    return {path.stem.lower() for path in root.rglob(f"*{suffix}") if path.is_file()}


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(items)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Check whether official MMP survival splits are covered by an existing WSI feature directory.")
    p.add_argument("--mmp-root", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--feature-suffix", default=".h5")
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--raw-svs-root", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    mmp_root = Path(args.mmp_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    if not feature_root.exists():
        raise FileNotFoundError(feature_root)

    feature_suffix = str(args.feature_suffix)
    if not feature_suffix.startswith("."):
        feature_suffix = f".{feature_suffix}"
    cohort = str(args.cohort).upper()

    split_root = (mmp_root / "src" / "splits" / "survival").resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (feature_root.parent / "readiness").resolve()
    raw_svs_root = Path(args.raw_svs_root).resolve() if args.raw_svs_root else None

    needed = _collect_needed_slides(split_root, cohort=cohort, n_folds=int(args.n_folds))
    needed_set = set(needed)
    feature_have = _index_stems(feature_root, feature_suffix)
    raw_have = _index_stems(raw_svs_root, ".svs") if raw_svs_root else set()

    missing_features = sorted(needed_set - feature_have)
    covered = sorted(needed_set & feature_have)
    raw_missing = sorted(needed_set - raw_have) if raw_svs_root else []

    _write_list(output_dir / "needed_slides.txt", needed)
    _write_list(output_dir / "covered_features.txt", covered)
    _write_list(output_dir / "missing_features.txt", missing_features)
    if raw_svs_root:
        _write_list(output_dir / "missing_raw_svs.txt", raw_missing)

    summary = {
        "cohort": cohort,
        "mmp_root": str(mmp_root),
        "feature_root": str(feature_root),
        "feature_suffix": feature_suffix,
        "need_total": len(needed),
        "feature_covered": len(covered),
        "feature_missing": len(missing_features),
        "raw_svs_root": str(raw_svs_root) if raw_svs_root else None,
        "raw_svs_missing": len(raw_missing) if raw_svs_root else None,
        "output_dir": str(output_dir),
        "examples": {
            "missing_features": missing_features[:20],
            "missing_raw_svs": raw_missing[:20] if raw_svs_root else [],
        },
    }

    summary_path = output_dir / "readiness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("need_total       =", summary["need_total"])
    print("feature_covered  =", summary["feature_covered"])
    print("feature_missing  =", summary["feature_missing"])
    if raw_svs_root:
        print("raw_svs_missing  =", summary["raw_svs_missing"])
    print("output_dir       =", output_dir)
    print("summary_json     =", summary_path)


if __name__ == "__main__":
    main()
