from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FoldRecord:
    fold: int
    best_val: float
    best_test: float
    final_test: float
    best_epoch: int
    reg_lambda: float


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(st.pstdev(values)) if len(values) > 1 else 0.0


def _parse_fold(path: Path) -> int:
    match = re.search(r"_k(\d+)_seed", path.parent.name)
    if match is None:
        raise RuntimeError(f"cannot parse fold from {path}")
    return int(match.group(1))


def _load_records(run_glob: str) -> list[FoldRecord]:
    records: list[FoldRecord] = []
    for summary_path in sorted(Path().glob(f"{run_glob}/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records.append(
            FoldRecord(
                fold=_parse_fold(summary_path),
                best_val=float(summary["best_val_c_index"]),
                best_test=float(summary["best_test_c_index"]),
                final_test=float(summary["final_test_c_index"]),
                best_epoch=int(summary["best_epoch"]),
                reg_lambda=float(summary.get("wsi_geo_reg_lambda", 0.0)),
            )
        )
    return sorted(records, key=lambda x: x.fold)


def _summarize(records: list[FoldRecord]) -> dict[str, float]:
    return {
        "best_val_mean": _mean([r.best_val for r in records]),
        "best_val_std": _std([r.best_val for r in records]),
        "best_test_mean": _mean([r.best_test for r in records]),
        "best_test_std": _std([r.best_test for r in records]),
        "final_test_mean": _mean([r.final_test for r in records]),
        "final_test_std": _std([r.final_test for r in records]),
        "best_epoch_mean": _mean([float(r.best_epoch) for r in records]),
        "lambda_mean": _mean([r.reg_lambda for r in records]),
    }


def _run_python(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args
    subprocess.run(cmd, check=True)


def _top_expression_features(path: Path, top_n: int = 6) -> list[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    return df["feature"].head(top_n).astype(str).tolist()


def _top_latent_dims(path: Path, top_n: int = 6) -> list[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    ranked = df.reindex(df["pos_minus_neg"].abs().sort_values(ascending=False).index).head(top_n)
    return [f"{int(row.dim)}({row.pos_minus_neg:+.2f})" for row in ranked.itertuples()]


def _plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    x_labels = summary_df["label"].tolist()
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(x, summary_df["best_test_mean"], marker="o", linewidth=2.0, label="geo best-test")
    axes[0].plot(x, summary_df["final_test_mean"], marker="s", linewidth=2.0, label="geo final-test")
    axes[0].axhline(float(summary_df["baseline_best_test_mean"].iloc[0]), color="black", linestyle="--", linewidth=1.0, label="baseline best-test")
    axes[0].axhline(float(summary_df["baseline_final_test_mean"].iloc[0]), color="gray", linestyle=":", linewidth=1.0, label="baseline final-test")
    axes[0].set_xticks(x, x_labels)
    axes[0].set_ylabel("C-index")
    axes[0].set_title("2D Laplacian Lambda vs No-Geo Baseline")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x, summary_df["net_rescue_pairs"], width=0.55, label="net rescue pairs")
    axes[1].plot(x, summary_df["delta_best_test_vs_baseline"], marker="o", linewidth=2.0, color="#d62728", label="delta best-test")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, x_labels)
    axes[1].set_title("Rescue / Harm Signal")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# 2D Laplacian Lambda Sweep vs No-Geo Baseline",
        "",
        "| label | lambda | geo best-test | delta vs baseline | geo final-test | net rescue | top WSI features | top fused dims |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_df.itertuples():
        lines.append(
            f"| {row.label} | {row.reg_lambda:.4f} | {row.best_test_mean:.4f} | {row.delta_best_test_vs_baseline:+.4f} | "
            f"{row.final_test_mean:.4f} | {int(row.net_rescue_pairs)} | {row.top_wsi_features} | {row.top_fused_dims} |"
        )

    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- best-test mean: {float(summary_df['baseline_best_test_mean'].iloc[0]):.4f}",
            f"- final-test mean: {float(summary_df['baseline_final_test_mean'].iloc[0]):.4f}",
            "",
            "## Readout",
            "",
            "- `top WSI features` comes from the strongest positive-vs-negative separation in `wsi_curve_*` diagnostics.",
            "- `top fused dims` lists latent dimensions with the largest `positive_mean_delta - negative_mean_delta`.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-compare 2D Laplacian WSI-geometry runs against the no-geo baseline.")
    parser.add_argument("--baseline_run_glob", required=True, help="Glob for baseline run directories, e.g. outputs/..._k*_seed0")
    parser.add_argument(
        "--geometry_run",
        action="append",
        default=[],
        help="Repeatable: label=glob for geometry run directories, e.g. lap0p01=outputs/..._k*_seed0",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if not args.geometry_run:
        raise RuntimeError("at least one --geometry_run label=glob is required")

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_records = _load_records(args.baseline_run_glob)
    if not baseline_records:
        raise RuntimeError("baseline summary.json files were not found")
    baseline_summary = _summarize(baseline_records)

    rescue_script = repo_root / "scripts" / "analyze_survival_rescue.py"
    spectra_script = repo_root / "scripts" / "compare_geo_baseline_latent_spectra.py"
    expr_script = repo_root / "scripts" / "analyze_geometry_expression_patterns.py"

    rows: list[dict[str, object]] = []
    for item in args.geometry_run:
        if "=" not in item:
            raise RuntimeError(f"invalid --geometry_run value: {item}")
        label, run_glob = item.split("=", 1)
        geom_records = _load_records(run_glob)
        if not geom_records:
            continue

        analysis_root = out_dir / label
        rescue_dir = analysis_root / "rescue"
        spectra_dir = analysis_root / "latent_spectra"
        expr_dir = analysis_root / "expression_wsi"
        analysis_root.mkdir(parents=True, exist_ok=True)

        baseline_cases = str((repo_root / f"{args.baseline_run_glob}/case_diagnostics/test_cases.csv").resolve())
        geom_cases = str((repo_root / f"{run_glob}/case_diagnostics/test_cases.csv").resolve())
        baseline_tensors = str((repo_root / f"{args.baseline_run_glob}/case_diagnostics/test_tensors.pt").resolve())
        geom_tensors = str((repo_root / f"{run_glob}/case_diagnostics/test_tensors.pt").resolve())

        _run_python(
            rescue_script,
            [
                "--baseline_glob",
                baseline_cases,
                "--geometry_glob",
                geom_cases,
                "--out_dir",
                str(rescue_dir),
            ],
        )
        _run_python(
            spectra_script,
            [
                "--baseline_glob",
                baseline_tensors,
                "--geometry_glob",
                geom_tensors,
                "--rescue_csv",
                str((rescue_dir / "rescue_case_ranking.csv").resolve()),
                "--out_dir",
                str(spectra_dir),
            ],
        )
        _run_python(
            expr_script,
            [
                "--rescue_csv",
                str((rescue_dir / "rescue_case_ranking.csv").resolve()),
                "--diag_glob",
                geom_cases,
                "--prefix",
                "wsi_curve",
                "--out_dir",
                str(expr_dir),
            ],
        )

        rescue_summary = json.loads((rescue_dir / "rescue_summary.json").read_text(encoding="utf-8"))
        geom_summary = _summarize(geom_records)
        rows.append(
            {
                "label": label,
                "reg_lambda": geom_summary["lambda_mean"],
                "best_test_mean": geom_summary["best_test_mean"],
                "final_test_mean": geom_summary["final_test_mean"],
                "delta_best_test_vs_baseline": geom_summary["best_test_mean"] - baseline_summary["best_test_mean"],
                "delta_final_test_vs_baseline": geom_summary["final_test_mean"] - baseline_summary["final_test_mean"],
                "net_rescue_pairs": int(rescue_summary["total_net_rescue_pairs"]),
                "rescue_pairs": int(rescue_summary["total_rescue_pairs"]),
                "harm_pairs": int(rescue_summary["total_harm_pairs"]),
                "top_wsi_features": ", ".join(_top_expression_features(expr_dir / "expression_feature_summary.csv")),
                "top_wsi_pathlike": ", ".join(
                    feat
                    for feat in _top_expression_features(expr_dir / "expression_feature_summary.csv", top_n=10)
                    if any(key in feat for key in ("path_length", "step_mean", "step_std", "step_max"))
                ),
                "top_fused_dims": ", ".join(_top_latent_dims(spectra_dir / "fused_delta_spectrum.csv")),
                "baseline_best_test_mean": baseline_summary["best_test_mean"],
                "baseline_final_test_mean": baseline_summary["final_test_mean"],
            }
        )

    if not rows:
        raise RuntimeError("no geometry runs were available for analysis")

    summary_df = pd.DataFrame(rows).sort_values(["reg_lambda", "label"], key=lambda x: pd.to_numeric(x, errors="coerce") if x.name == "reg_lambda" else x)
    summary_df.to_csv(out_dir / "lambda_vs_nogeo_summary.csv", index=False)
    _plot_summary(summary_df, out_dir / "lambda_vs_nogeo_summary.png")
    _write_markdown(summary_df, out_dir / "lambda_vs_nogeo_summary.md")

    best_row = summary_df.iloc[int(np.nanargmax(summary_df["best_test_mean"].to_numpy(dtype=float)))]
    meta = {
        "baseline_best_test_mean": baseline_summary["best_test_mean"],
        "baseline_final_test_mean": baseline_summary["final_test_mean"],
        "best_lambda_by_best_test": {
            "label": str(best_row["label"]),
            "lambda": float(best_row["reg_lambda"]),
            "best_test_mean": float(best_row["best_test_mean"]),
            "delta_vs_baseline": float(best_row["delta_best_test_vs_baseline"]),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
