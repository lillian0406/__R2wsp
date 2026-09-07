from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FoldRecord:
    fold: int
    best_val: float
    best_test: float
    final_test: float
    best_epoch: int


def _load_records(pattern: str) -> list[FoldRecord]:
    records: list[FoldRecord] = []
    for summary_path in sorted(Path("/root/autodl-tmp/R2wsp/outputs").glob(pattern)):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        match = re.search(r"_k(\d+)_seed", summary_path.parent.name)
        if match is None:
            continue
        records.append(
            FoldRecord(
                fold=int(match.group(1)),
                best_val=float(summary["best_val_c_index"]),
                best_test=float(summary["best_test_c_index"]),
                final_test=float(summary["final_test_c_index"]),
                best_epoch=int(summary["best_epoch"]),
            )
        )
    return sorted(records, key=lambda x: x.fold)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(st.pstdev(values)) if len(values) > 1 else 0.0


def _summarize(records: list[FoldRecord]) -> dict[str, float]:
    best_val = [r.best_val for r in records]
    best_test = [r.best_test for r in records]
    final_test = [r.final_test for r in records]
    best_epoch = [r.best_epoch for r in records]
    return {
        "best_val_mean": _mean(best_val),
        "best_val_std": _std(best_val),
        "best_test_mean": _mean(best_test),
        "best_test_std": _std(best_test),
        "final_test_mean": _mean(final_test),
        "final_test_std": _std(final_test),
        "best_epoch_mean": _mean(best_epoch),
    }


def _plot_mean_summary(configs: dict[str, list[FoldRecord]], out_path: Path) -> None:
    labels = list(configs.keys())
    best_test_mean = [_summarize(configs[k])["best_test_mean"] for k in labels]
    best_test_std = [_summarize(configs[k])["best_test_std"] for k in labels]
    final_test_mean = [_summarize(configs[k])["final_test_mean"] for k in labels]
    final_test_std = [_summarize(configs[k])["final_test_std"] for k in labels]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(13, 5.5))
    plt.bar(x - width / 2, best_test_mean, width=width, yerr=best_test_std, capsize=4, label="best-val selected test")
    plt.bar(x + width / 2, final_test_mean, width=width, yerr=final_test_std, capsize=4, label="final_test")
    plt.xticks(x, labels)
    plt.ylim(0.48, 0.68)
    plt.ylabel("C-index")
    plt.title("WSI Geometry Dim Comparison on DINO+Across+OmicsAttn")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_fold_curves(configs: dict[str, list[FoldRecord]], out_path: Path) -> None:
    baseline = {r.fold: r for r in configs["baseline_3d_noreg"]}
    colors = {
        "baseline_3d_noreg": "#555555",
        "wsi2d_lap0p05": "#1f77b4",
        "wsi3d_lap0p05": "#ff7f0e",
        "wsi4d_lap0p05": "#2ca02c",
        "wsi3d_lap_match": "#d62728",
        "wsi4d_lap_match": "#9467bd",
    }
    display = {
        "baseline_3d_noreg": "baseline 3D no-reg",
        "wsi2d_lap0p05": "2D + Lap 0.05",
        "wsi3d_lap0p05": "3D + Lap 0.05",
        "wsi4d_lap0p05": "4D + Lap 0.05",
        "wsi3d_lap_match": "3D + Lap match",
        "wsi4d_lap_match": "4D + Lap match",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for key, records in configs.items():
        folds = [r.fold for r in records]
        axes[0].plot(folds, [r.best_test for r in records], marker="o", linewidth=1.8, label=display[key], color=colors[key])
    axes[0].set_title("Per-fold best-val selected test")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("C-index")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    for key in ("wsi2d_lap0p05", "wsi3d_lap0p05", "wsi4d_lap0p05", "wsi3d_lap_match", "wsi4d_lap_match"):
        records = configs[key]
        folds = [r.fold for r in records]
        delta = [r.best_test - baseline[r.fold].best_test for r in records]
        axes[1].plot(folds, delta, marker="o", linewidth=1.8, label=display[key], color=colors[key])
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Delta vs baseline (best-val selected test)")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Delta C-index")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_match_focus(configs: dict[str, list[FoldRecord]], out_path: Path) -> None:
    summaries = {k: _summarize(v) for k, v in configs.items()}
    labels = ["baseline_3d_noreg", "wsi2d_lap0p05", "wsi3d_lap0p05", "wsi3d_lap_match", "wsi4d_lap0p05", "wsi4d_lap_match"]
    display = ["baseline", "2D+0.05", "3D+0.05", "3D+match", "4D+0.05", "4D+match"]
    best_test_mean = [summaries[k]["best_test_mean"] for k in labels]
    final_test_mean = [summaries[k]["final_test_mean"] for k in labels]

    x = np.arange(len(labels))
    plt.figure(figsize=(11, 4.8))
    plt.plot(x, best_test_mean, marker="o", linewidth=2.0, label="best-val selected test")
    plt.plot(x, final_test_mean, marker="s", linewidth=2.0, label="final_test")
    plt.xticks(x, display)
    plt.ylim(0.56, 0.65)
    plt.ylabel("C-index")
    plt.title("Matched-strength Focus: 3D/4D Laplacian vs Baseline")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_markdown(configs: dict[str, list[FoldRecord]], out_path: Path) -> None:
    summaries = {k: _summarize(v) for k, v in configs.items()}
    baseline = summaries["baseline_3d_noreg"]
    lines = [
        "# Geometry Dimension Comparison",
        "",
        "| config | best_val_mean | best-test mean | final_test_mean | best_epoch_mean | delta best-test vs baseline | delta final vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, s in summaries.items():
        lines.append(
            f"| {key} | {s['best_val_mean']:.4f} | {s['best_test_mean']:.4f} | {s['final_test_mean']:.4f} | {s['best_epoch_mean']:.1f} | "
            f"{s['best_test_mean'] - baseline['best_test_mean']:+.4f} | {s['final_test_mean'] - baseline['final_test_mean']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-fold best-test",
            "",
            "| fold | baseline_3d_noreg | wsi2d_lap0p05 | wsi3d_lap0p05 | wsi4d_lap0p05 | wsi3d_lap_match | wsi4d_lap_match |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_key = {k: {r.fold: r for r in v} for k, v in configs.items()}
    for fold in range(5):
        lines.append(
            f"| {fold} | {by_key['baseline_3d_noreg'][fold].best_test:.4f} | {by_key['wsi2d_lap0p05'][fold].best_test:.4f} | "
            f"{by_key['wsi3d_lap0p05'][fold].best_test:.4f} | {by_key['wsi4d_lap0p05'][fold].best_test:.4f} | "
            f"{by_key['wsi3d_lap_match'][fold].best_test:.4f} | {by_key['wsi4d_lap_match'][fold].best_test:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/root/autodl-tmp/R2wsp/outputs/paper_geometry_analysis/geo_dim_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline_3d_noreg": _load_records("_push62_dino_across_omicsattn_k*_seed0/summary.json"),
        "wsi2d_lap0p05": _load_records("_geoexp_dino_across_omicsattn_wsi2d_lap5e2_k*_seed0/summary.json"),
        "wsi3d_lap0p05": _load_records("_geoexp_dino_across_omicsattn_wsi3d_lap5e2_k*_seed0/summary.json"),
        "wsi4d_lap0p05": _load_records("_geoexp_dino_across_omicsattn_wsi4d_lap5e2_k*_seed0/summary.json"),
        "wsi3d_lap_match": _load_records("_geoexp_dino_across_omicsattn_wsi3d_lap3p7e2match_k*_seed0/summary.json"),
        "wsi4d_lap_match": _load_records("_geoexp_dino_across_omicsattn_wsi4d_lap3p0e2match_k*_seed0/summary.json"),
    }
    _plot_mean_summary(configs, out_dir / "geo_dim_mean_summary.png")
    _plot_fold_curves(configs, out_dir / "geo_dim_fold_curves.png")
    _plot_match_focus(configs, out_dir / "geo_dim_match_focus.png")
    _write_markdown(configs, out_dir / "geo_dim_summary.md")


if __name__ == "__main__":
    main()
