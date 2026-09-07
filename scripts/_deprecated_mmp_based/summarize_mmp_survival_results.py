from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.eval.mmp_survival_summary import collect_results, summarize_results, write_summary_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize MMP survival 5-fold results and check whether they meet the target baseline."
    )
    p.add_argument(
        "--results-root",
        default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_plip_luad_256_official_dss/results_5fold",
        help="Directory that contains k=0 ... k=4 result folders.",
    )
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--min-mean-cindex", type=float, default=0.64)
    p.add_argument("--max-std", type=float, default=0.04)
    p.add_argument("--max-logrank-p", type=float, default=0.05)
    p.add_argument("--target-mean-cindex", type=float, default=0.665)
    p.add_argument("--target-mean-tolerance", type=float, default=0.025)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(results_root)

    metrics = collect_results(results_root)
    summary = summarize_results(
        metrics,
        expected_folds=int(args.expected_folds),
        min_mean_cindex=float(args.min_mean_cindex),
        max_std=float(args.max_std),
        max_logrank_p=float(args.max_logrank_p),
        target_mean_cindex=float(args.target_mean_cindex),
        target_mean_tolerance=float(args.target_mean_tolerance),
    )

    output_json = Path(args.output_json).resolve() if args.output_json else (results_root / "aggregate_summary.json")
    output_csv = Path(args.output_csv).resolve() if args.output_csv else (results_root / "aggregate_summary.csv")

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(output_csv, metrics, summary)

    print("status       =", summary["status"])
    print("results_root =", results_root)
    print("folds_found  =", summary["folds_found"])
    print("mean_c_index =", summary["mean_c_index"])
    print("std_c_index  =", summary["std_c_index"])
    print("max_logrank_p=", summary["max_logrank_p"])
    print("checks       =", json.dumps(summary["checks"], ensure_ascii=False, sort_keys=True))
    print("output_json  =", output_json)
    print("output_csv   =", output_csv)

    phase2 = summary.get("best_fold_for_phase2", {})
    print("phase2_fold  =", phase2.get("fold"))
    print("phase2_cidx  =", phase2.get("c_index"))
    print("phase2_dump  =", phase2.get("dump_path"))
    print("phase2_log   =", phase2.get("log_path"))


if __name__ == "__main__":
    main()
