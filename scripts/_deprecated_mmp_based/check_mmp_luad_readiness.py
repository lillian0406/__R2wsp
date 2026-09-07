from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import resolve_data_paths


def _collect_needed_slides(split_root: Path, cohort: str, n_folds: int) -> list[str]:
    needed: set[str] = set()
    for k in range(int(n_folds)):
        fold_dir = split_root / f"TCGA_{cohort}_overall_survival_k={k}"
        for split_name in ["train.csv", "test.csv"]:
            split_path = fold_dir / split_name
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            df = pd.read_csv(split_path)
            needed.update(df["slide_id"].astype(str).str.lower().tolist())
    return sorted(needed)


def _index_stems(root: Path, suffix: str) -> set[str]:
    if not root.exists():
        return set()
    return {p.stem.lower() for p in root.rglob(f"*{suffix}") if p.is_file()}


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(items)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Check whether LUAD official-split assets are ready for the next MMP baseline run.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--mmp-root", default="/root/autodl-tmp/_refs/MMP-main")
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--token-source", default="plip_luad_256")
    p.add_argument("--feature-name", default="plip_luad_256")
    p.add_argument("--patch-mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    cohort = str(args.cohort).upper()
    data_root = paths.data_root.resolve()
    mmp_root = Path(args.mmp_root).resolve()
    split_root = (mmp_root / "src" / "splits" / "survival").resolve()
    raw_svs_dir = (data_root / "raw_svs").resolve()
    token_dir = (data_root / "tokens" / args.token_source).resolve()
    bridge_root = (
        Path(args.bridge_root).resolve()
        if args.bridge_root
        else (data_root / "bridges" / f"mmp_{cohort.lower()}_{args.token_source}_official_dss").resolve()
    )
    feats_pt_dir = (
        bridge_root
        / "histology"
        / f"extracted_mag{int(args.patch_mag)}x_patch{int(args.patch_size)}_fp"
        / args.feature_name
        / "feats_pt"
    ).resolve()
    results_root = Path(args.results_root).resolve() if args.results_root else (bridge_root / "results_5fold").resolve()
    readiness_dir = (bridge_root / "readiness").resolve()

    needed = _collect_needed_slides(split_root, cohort=cohort, n_folds=int(args.n_folds))
    needed_set = set(needed)
    raw_have = _index_stems(raw_svs_dir, ".svs")
    token_have = _index_stems(token_dir, ".npz")
    feats_pt_have = _index_stems(feats_pt_dir, ".pt")

    raw_missing = sorted(needed_set - raw_have)
    token_missing = sorted(needed_set - token_have)
    feats_pt_missing = sorted(needed_set - feats_pt_have)
    ready_for_tokenization = sorted((needed_set & raw_have) - token_have)
    ready_for_bridge_refresh = sorted((needed_set & token_have) - feats_pt_have)

    _write_list(readiness_dir / "needed_slides.txt", needed)
    _write_list(readiness_dir / "missing_raw_svs.txt", raw_missing)
    _write_list(readiness_dir / "missing_tokens.txt", token_missing)
    _write_list(readiness_dir / "missing_feats_pt.txt", feats_pt_missing)
    _write_list(readiness_dir / "ready_for_tokenization.txt", ready_for_tokenization)
    _write_list(readiness_dir / "ready_for_bridge_refresh.txt", ready_for_bridge_refresh)

    next_step: str
    next_command: str | None = None
    if raw_missing:
        next_step = "continue_download_raw_svs"
        next_command = (
            "python scripts/check_mmp_luad_readiness.py "
            f"--mmp-root {mmp_root} --token-source {args.token_source}"
        )
    elif token_missing:
        next_step = "run_tokenization_for_missing_slides"
    elif feats_pt_missing:
        next_step = "refresh_bridge_feats_pt"
        next_command = (
            "python scripts/prepare_mmp_official_luad_plip.py "
            f"--mmp_root {mmp_root} --token_source {args.token_source} --feature_name {args.feature_name}"
        )
    else:
        next_step = "ready_to_rerun_baseline_v2"
        next_command = (
            "python scripts/run_mmp_official_luad_plip_5fold.py "
            f"--mmp-root {mmp_root} --bridge-root {bridge_root} --python-executable /root/miniconda3/envs/mmp/bin/python"
        )

    summary = {
        "cohort": cohort,
        "need_total": len(needed),
        "raw_svs_dir": str(raw_svs_dir),
        "token_dir": str(token_dir),
        "feats_pt_dir": str(feats_pt_dir),
        "bridge_root": str(bridge_root),
        "results_root": str(results_root),
        "raw_svs_covered": len(needed_set & raw_have),
        "raw_svs_missing": len(raw_missing),
        "token_covered": len(needed_set & token_have),
        "token_missing": len(token_missing),
        "feats_pt_covered": len(needed_set & feats_pt_have),
        "feats_pt_missing": len(feats_pt_missing),
        "ready_for_tokenization": len(ready_for_tokenization),
        "ready_for_bridge_refresh": len(ready_for_bridge_refresh),
        "next_step": next_step,
        "next_command": next_command,
        "readiness_dir": str(readiness_dir),
        "examples": {
            "missing_raw_svs": raw_missing[:20],
            "missing_tokens": token_missing[:20],
            "missing_feats_pt": feats_pt_missing[:20],
            "ready_for_tokenization": ready_for_tokenization[:20],
            "ready_for_bridge_refresh": ready_for_bridge_refresh[:20],
        },
    }

    output_json = Path(args.output_json).resolve() if args.output_json else (readiness_dir / "readiness_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("need_total              =", summary["need_total"])
    print("raw_svs_covered         =", summary["raw_svs_covered"])
    print("raw_svs_missing         =", summary["raw_svs_missing"])
    print("token_covered           =", summary["token_covered"])
    print("token_missing           =", summary["token_missing"])
    print("feats_pt_covered        =", summary["feats_pt_covered"])
    print("feats_pt_missing        =", summary["feats_pt_missing"])
    print("ready_for_tokenization  =", summary["ready_for_tokenization"])
    print("ready_for_bridge_refresh=", summary["ready_for_bridge_refresh"])
    print("next_step               =", next_step)
    print("next_command            =", next_command)
    print("output_json             =", output_json)
    print("readiness_dir           =", readiness_dir)


if __name__ == "__main__":
    main()
