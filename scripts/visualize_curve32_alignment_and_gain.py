from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _load_run_case(run_dir: Path, case_id: str, prefix: str) -> dict[str, object] | None:
    tensor_path = run_dir / "case_diagnostics" / "test_tensors.pt"
    case_csv = run_dir / "case_diagnostics" / "test_cases.csv"
    if not tensor_path.exists() or not case_csv.exists():
        return None
    tensors = torch.load(tensor_path, map_location="cpu")
    case_df = pd.read_csv(case_csv)
    all_case_ids = [str(x) for x in tensors["case_id"]]
    try:
        idx = all_case_ids.index(str(case_id))
    except ValueError:
        return None
    row = case_df.loc[case_df["case_id"] == case_id]
    if row.empty:
        return None
    item = row.iloc[0].to_dict()
    out: dict[str, object] = {
        "run_name": run_dir.name,
        "case_id": str(case_id),
        "meta": item,
        "points": tensors[f"{prefix}_points"][idx].cpu().numpy(),
    }
    curv_key = f"{prefix}_curvature_per_point"
    tors_key = f"{prefix}_torsion_per_point"
    out["curvature"] = tensors[curv_key][idx].cpu().numpy() if curv_key in tensors else None
    out["torsion"] = tensors[tors_key][idx].cpu().numpy() if tors_key in tensors else None
    return out


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    scale = float(np.linalg.norm(centered))
    if scale <= 1e-8:
        scale = 1.0
    return centered / scale, center, scale


def _kabsch_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h = src.T @ ref
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return src @ r


def _align_to_reference(point_sets: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
    normed = [_normalize_points(p)[0] for p in point_sets]
    ref = normed[0]
    aligned = [ref]
    rmsd_values = [0.0]
    for points in normed[1:]:
        ali = _kabsch_align(points, ref)
        aligned.append(ali)
        rmsd_values.append(float(np.sqrt(np.mean((ali - ref) ** 2))))
    stats = {
        "mean_rmsd_to_ref": float(np.mean(rmsd_values)),
        "max_rmsd_to_ref": float(np.max(rmsd_values)),
    }
    return aligned, stats


def _curve_stats(points: np.ndarray) -> dict[str, np.ndarray | float]:
    deltas = points[1:] - points[:-1]
    step = np.linalg.norm(deltas, axis=1)
    accel = deltas[1:] - deltas[:-1] if len(deltas) >= 2 else np.zeros((0, 3), dtype=np.float64)
    accel_norm = np.linalg.norm(accel, axis=1)
    return {
        "step": step,
        "accel_norm": accel_norm,
        "path_length": float(step.sum()),
        "endpoint_disp": float(np.linalg.norm(points[-1] - points[0])),
    }


def _plot_overlay(ax, curves: list[np.ndarray], dims: tuple[int, int], labels: list[str], title: str) -> None:
    cmap = plt.cm.tab10(np.linspace(0.0, 1.0, len(curves)))
    for color, curve, label in zip(cmap, curves, labels):
        ax.plot(curve[:, dims[0]], curve[:, dims[1]], linewidth=1.4, alpha=0.85, color=color, label=f"run {label}")
        ax.scatter(curve[:, dims[0]], curve[:, dims[1]], s=10, color=color, alpha=0.7)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _plot_gain_curve(ax, points: np.ndarray, title: str) -> None:
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(points)))
    ax.plot(points[:, 0], points[:, 1], linewidth=1.5, color="#3366cc")
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=18)
    ax.scatter(points[0, 0], points[0, 1], c="green", s=44)
    ax.scatter(points[-1, 0], points[-1, 1], c="red", s=44)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def make_stability_figure(run_dirs: list[Path], case_id: str, prefix: str, out_path: Path) -> None:
    items = []
    for run_dir in run_dirs:
        item = _load_run_case(run_dir, case_id, prefix)
        if item is not None:
            items.append(item)
    if len(items) < 2:
        raise RuntimeError("need at least two runs containing the same case to draw stability figure")

    point_sets = [item["points"] for item in items]  # type: ignore[index]
    aligned, align_stats = _align_to_reference(point_sets)
    labels = [str(item["run_name"]) for item in items]
    step_series = np.stack([_curve_stats(c)["step"] for c in aligned], axis=0)
    accel_series = np.stack([_curve_stats(c)["accel_norm"] for c in aligned], axis=0)
    spread = np.std(np.stack(aligned, axis=0), axis=0).mean(axis=1)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_step = fig.add_subplot(gs[1, 0])
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_spread = fig.add_subplot(gs[1, 2])

    _plot_overlay(ax_xy, aligned, (0, 1), labels, "Aligned XY Overlay")
    _plot_overlay(ax_xz, aligned, (0, 2), labels, "Aligned XZ Overlay")
    _plot_overlay(ax_yz, aligned, (1, 2), labels, "Aligned YZ Overlay")
    ax_xy.legend(fontsize=8)

    x_step = np.arange(1, step_series.shape[1] + 1)
    ax_step.plot(x_step, step_series.T, alpha=0.35)
    ax_step.plot(x_step, step_series.mean(axis=0), color="black", linewidth=2.0, label="mean")
    ax_step.fill_between(x_step, step_series.mean(axis=0) - step_series.std(axis=0), step_series.mean(axis=0) + step_series.std(axis=0), alpha=0.2)
    ax_step.set_title("Step-Length Stability")
    ax_step.grid(alpha=0.25)
    ax_step.legend(fontsize=8)

    x_accel = np.arange(2, 2 + accel_series.shape[1])
    ax_accel.plot(x_accel, accel_series.T, alpha=0.35)
    ax_accel.plot(x_accel, accel_series.mean(axis=0), color="black", linewidth=2.0, label="mean")
    ax_accel.fill_between(x_accel, accel_series.mean(axis=0) - accel_series.std(axis=0), accel_series.mean(axis=0) + accel_series.std(axis=0), alpha=0.2)
    ax_accel.set_title("Smoothness Stability")
    ax_accel.grid(alpha=0.25)
    ax_accel.legend(fontsize=8)

    ax_spread.plot(np.arange(1, len(spread) + 1), spread, marker="o", linewidth=1.5)
    ax_spread.set_title("Pointwise Spread After Alignment")
    ax_spread.grid(alpha=0.25)

    summary = {
        "case_id": case_id,
        "prefix": prefix,
        "runs": labels,
        "align_stats": align_stats,
        "path_length_mean": float(np.mean([_curve_stats(c)["path_length"] for c in aligned])),
        "path_length_std": float(np.std([_curve_stats(c)["path_length"] for c in aligned])),
        "endpoint_disp_mean": float(np.mean([_curve_stats(c)["endpoint_disp"] for c in aligned])),
        "endpoint_disp_std": float(np.std([_curve_stats(c)["endpoint_disp"] for c in aligned])),
    }
    fig.suptitle(f"Curve32 Stability Across Runs: {case_id} | {prefix}", fontsize=15)
    fig.text(0.50, 0.02, json.dumps(summary, ensure_ascii=False, indent=2), ha="center", va="bottom", fontsize=9, family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (out_path.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def make_gain_compare_figure(
    pos_run_dir: Path,
    pos_case_id: str,
    neg_run_dir: Path,
    neg_case_id: str,
    prefix: str,
    out_path: Path,
) -> None:
    pos_item = _load_run_case(pos_run_dir, pos_case_id, prefix)
    neg_item = _load_run_case(neg_run_dir, neg_case_id, prefix)
    if pos_item is None or neg_item is None:
        raise RuntimeError("positive or negative case not found in provided runs")

    pos_points = pos_item["points"]  # type: ignore[index]
    neg_points = neg_item["points"]  # type: ignore[index]
    pos_curv = pos_item["curvature"]  # type: ignore[index]
    neg_curv = neg_item["curvature"]  # type: ignore[index]
    pos_stats = _curve_stats(pos_points)
    neg_stats = _curve_stats(neg_points)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    _plot_gain_curve(ax1, pos_points, f"Positive Gain: {pos_case_id}")
    _plot_gain_curve(ax2, neg_points, f"Negative Gain: {neg_case_id}")
    _plot_overlay(ax3, [_normalize_points(pos_points)[0], _normalize_points(neg_points)[0]], (0, 1), ["positive", "negative"], "Normalized XY Compare")
    ax3.legend(fontsize=8)

    x_step_pos = np.arange(1, len(pos_stats["step"]) + 1)
    x_step_neg = np.arange(1, len(neg_stats["step"]) + 1)
    ax4.plot(x_step_pos, pos_stats["step"], label="positive", linewidth=1.5)
    ax4.plot(x_step_neg, neg_stats["step"], label="negative", linewidth=1.5)
    ax4.set_title("Step Length")
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8)

    x_acc_pos = np.arange(2, 2 + len(pos_stats["accel_norm"]))
    x_acc_neg = np.arange(2, 2 + len(neg_stats["accel_norm"]))
    ax5.plot(x_acc_pos, pos_stats["accel_norm"], label="positive", linewidth=1.5)
    ax5.plot(x_acc_neg, neg_stats["accel_norm"], label="negative", linewidth=1.5)
    ax5.set_title("2nd-diff Norm")
    ax5.grid(alpha=0.25)
    ax5.legend(fontsize=8)

    if pos_curv is not None:
        ax6.plot(np.arange(1, len(pos_curv) + 1), pos_curv, label="positive", linewidth=1.5)
    if neg_curv is not None:
        ax6.plot(np.arange(1, len(neg_curv) + 1), neg_curv, label="negative", linewidth=1.5)
    ax6.set_title("Curvature")
    ax6.grid(alpha=0.25)
    ax6.legend(fontsize=8)

    summary = {
        "prefix": prefix,
        "positive_case": pos_case_id,
        "negative_case": neg_case_id,
        "positive_path_length": float(pos_stats["path_length"]),
        "negative_path_length": float(neg_stats["path_length"]),
        "positive_endpoint_disp": float(pos_stats["endpoint_disp"]),
        "negative_endpoint_disp": float(neg_stats["endpoint_disp"]),
        "positive_meta": pos_item["meta"],
        "negative_meta": neg_item["meta"],
    }
    fig.suptitle(f"Curve32 Positive-vs-Negative Gain Comparison | {prefix}", fontsize=15)
    fig.text(0.50, 0.02, json.dumps(summary, ensure_ascii=False, indent=2), ha="center", va="bottom", fontsize=9, family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (out_path.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize curve32 stability across runs and positive-vs-negative gain comparisons.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    stab = subparsers.add_parser("stability")
    stab.add_argument("--run_dirs", required=True, help="Comma-separated run directories with case_diagnostics")
    stab.add_argument("--case_id", required=True)
    stab.add_argument("--prefix", required=True, choices=["wsi_curve", "rna_curve"])
    stab.add_argument("--out_path", required=True)

    gain = subparsers.add_parser("gain_compare")
    gain.add_argument("--pos_run_dir", required=True)
    gain.add_argument("--pos_case_id", required=True)
    gain.add_argument("--neg_run_dir", required=True)
    gain.add_argument("--neg_case_id", required=True)
    gain.add_argument("--prefix", required=True, choices=["wsi_curve", "rna_curve"])
    gain.add_argument("--out_path", required=True)

    args = parser.parse_args()
    if args.mode == "stability":
        run_dirs = [Path(x.strip()).resolve() for x in str(args.run_dirs).split(",") if x.strip()]
        make_stability_figure(run_dirs, args.case_id, args.prefix, Path(args.out_path).resolve())
    else:
        make_gain_compare_figure(
            Path(args.pos_run_dir).resolve(),
            args.pos_case_id,
            Path(args.neg_run_dir).resolve(),
            args.neg_case_id,
            args.prefix,
            Path(args.out_path).resolve(),
        )


if __name__ == "__main__":
    main()
