from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np


def _parse_fold(run_name: str, summary: dict[str, object]) -> int | None:
    split_dir = str(summary.get("split_dir", ""))
    match = re.search(r"k=(\d+)", split_dir)
    if match:
        return int(match.group(1))
    match = re.search(r"_k(\d+)_", run_name)
    return int(match.group(1)) if match else None


def _parse_seed(run_name: str, summary: dict[str, object]) -> int | None:
    seed = summary.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    match = re.search(r"seed(\d+)", run_name)
    return int(match.group(1)) if match else None


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize strict matched ablation results from run summary.json files.")
    parser.add_argument("--group", action="append", required=True, help="Group spec: label=/abs/path/pattern/to/summary.json")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in args.group:
        if "=" not in spec:
            raise ValueError(f"invalid group spec: {spec}")
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            summary_rows.append(
                {
                    "group": label,
                    "n_runs": 0,
                    "n_folds": 0,
                    "n_seeds": 0,
                    "best_val_mean": None,
                    "best_val_std": None,
                    "best_test_mean": None,
                    "best_test_std": None,
                    "final_test_mean": None,
                    "final_test_std": None,
                }
            )
            continue

        runs: list[dict[str, object]] = []
        for path_str in paths:
            path = Path(path_str)
            summary = json.loads(path.read_text(encoding="utf-8"))
            run_name = path.parent.name
            row = {
                "group": label,
                "run_name": run_name,
                "run_dir": str(path.parent),
                "fold": _parse_fold(run_name, summary),
                "seed": _parse_seed(run_name, summary),
                "best_val_c_index": summary.get("best_val_c_index"),
                "best_val_epoch": summary.get("best_epoch"),
                "best_test_c_index": summary.get("best_test_c_index"),
                "final_test_c_index": summary.get("final_test_c_index"),
                "cross_modal_fusion": summary.get("cross_modal_fusion"),
                "wsi_feature_source": summary.get("wsi_feature_source"),
                "wsi_geo_type": summary.get("wsi_geo_type"),
                "rna_mode": summary.get("rna_mode"),
                "rna_geo_type": summary.get("rna_geo_type"),
                "batch_size": summary.get("batch_size"),
            }
            detail_rows.append(row)
            runs.append(row)

        best_vals = [float(x["best_val_c_index"]) for x in runs if x["best_val_c_index"] is not None]
        best_tests = [float(x["best_test_c_index"]) for x in runs if x["best_test_c_index"] is not None]
        final_tests = [float(x["final_test_c_index"]) for x in runs if x["final_test_c_index"] is not None]
        best_val_mean, best_val_std = _mean_std(best_vals)
        best_test_mean, best_test_std = _mean_std(best_tests)
        final_test_mean, final_test_std = _mean_std(final_tests)
        summary_rows.append(
            {
                "group": label,
                "n_runs": len(runs),
                "n_folds": len({x["fold"] for x in runs}),
                "n_seeds": len({x["seed"] for x in runs}),
                "best_val_mean": best_val_mean,
                "best_val_std": best_val_std,
                "best_test_mean": best_test_mean,
                "best_test_std": best_test_std,
                "final_test_mean": final_test_mean,
                "final_test_std": final_test_std,
            }
        )

    out_dir = Path(args.out_dir).resolve()
    _write_csv(
        out_dir / "matched_ablation_detail.csv",
        detail_rows,
        [
            "group",
            "run_name",
            "run_dir",
            "fold",
            "seed",
            "best_val_c_index",
            "best_val_epoch",
            "best_test_c_index",
            "final_test_c_index",
            "cross_modal_fusion",
            "wsi_feature_source",
            "wsi_geo_type",
            "rna_mode",
            "rna_geo_type",
            "batch_size",
        ],
    )
    _write_csv(
        out_dir / "matched_ablation_summary.csv",
        summary_rows,
        [
            "group",
            "n_runs",
            "n_folds",
            "n_seeds",
            "best_val_mean",
            "best_val_std",
            "best_test_mean",
            "best_test_std",
            "final_test_mean",
            "final_test_std",
        ],
    )

    md_lines = [
        "# Matched Ablation Summary",
        "",
        "| group | n_runs | n_folds | n_seeds | best_val_mean | best-val selected test mean | final_test_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {group} | {n_runs} | {n_folds} | {n_seeds} | {best_val_mean} | {best_test_mean} | {final_test_mean} |".format(
                group=row["group"],
                n_runs=row["n_runs"],
                n_folds=row["n_folds"],
                n_seeds=row["n_seeds"],
                best_val_mean="-" if row["best_val_mean"] is None else f"{float(row['best_val_mean']):.4f}",
                best_test_mean="-" if row["best_test_mean"] is None else f"{float(row['best_test_mean']):.4f}",
                final_test_mean="-" if row["final_test_mean"] is None else f"{float(row['final_test_mean']):.4f}",
            )
        )
    (out_dir / "matched_ablation_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
