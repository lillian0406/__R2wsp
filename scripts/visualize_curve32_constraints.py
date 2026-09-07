from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _resolve_case_ids(args: argparse.Namespace) -> list[str]:
    if args.case_ids:
        return [x.strip() for x in str(args.case_ids).split(",") if x.strip()]
    if not args.case_table:
        raise ValueError("either --case_ids or --case_table must be provided")
    df = pd.read_csv(args.case_table)
    return [str(x) for x in df["case_id"].head(int(args.top_n)).tolist()]


def _case_index(case_ids: list[str], target: str) -> int | None:
    for idx, case_id in enumerate(case_ids):
        if str(case_id) == str(target):
            return idx
    return None


def _compute_curve_stats(points: np.ndarray) -> dict[str, float | np.ndarray]:
    deltas = points[1:] - points[:-1]
    step_norm = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_norm)])
    accel = deltas[1:] - deltas[:-1] if len(deltas) >= 2 else np.zeros((0, 3), dtype=np.float64)
    accel_norm = np.linalg.norm(accel, axis=1)
    cos_vals = []
    for i in range(len(deltas) - 1):
        a = deltas[i]
        b = deltas[i + 1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-8:
            cos_vals.append(1.0)
        else:
            cos_vals.append(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)))
    cos_vals = np.asarray(cos_vals, dtype=np.float64)
    return {
        "step_norm": step_norm,
        "cumulative": cumulative,
        "accel_norm": accel_norm,
        "turn_cos": cos_vals,
        "path_length": float(step_norm.sum()),
        "endpoint_disp": float(np.linalg.norm(points[-1] - points[0])),
        "step_mean": float(step_norm.mean()) if len(step_norm) else 0.0,
        "step_std": float(step_norm.std()) if len(step_norm) else 0.0,
        "accel_mean": float(accel_norm.mean()) if len(accel_norm) else 0.0,
        "accel_max": float(accel_norm.max()) if len(accel_norm) else 0.0,
        "turn_cos_mean": float(cos_vals.mean()) if len(cos_vals) else 1.0,
    }


def _plot_projection(ax, points: np.ndarray, dims: tuple[int, int], title: str) -> None:
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(points)))
    ax.plot(points[:, dims[0]], points[:, dims[1]], color="#3366cc", linewidth=1.5, alpha=0.8)
    ax.scatter(points[:, dims[0]], points[:, dims[1]], c=colors, s=20)
    for i in range(0, len(points), max(1, len(points) // 8)):
        ax.text(points[i, dims[0]], points[i, dims[1]], str(i + 1), fontsize=7)
    ax.scatter(points[0, dims[0]], points[0, dims[1]], c="green", s=48, label="start")
    ax.scatter(points[-1, dims[0]], points[-1, dims[1]], c="red", s=48, label="end")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _plot_3d(ax, points: np.ndarray) -> None:
    colors = plt.cm.plasma(np.linspace(0.0, 1.0, len(points)))
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#7b1fa2", linewidth=1.4, alpha=0.8)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=18)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], c="green", s=46)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="red", s=46)
    ax.set_title("3D Ordered Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _plot_curve_constraints(
    *,
    case_id: str,
    prefix: str,
    points: np.ndarray,
    curvature: np.ndarray | None,
    torsion: np.ndarray | None,
    meta: dict[str, float | int | str],
    out_path: Path,
) -> None:
    stats = _compute_curve_stats(points)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")
    ax_step = fig.add_subplot(gs[1, 0])
    ax_smooth = fig.add_subplot(gs[1, 1])
    ax_curv = fig.add_subplot(gs[1, 2])

    _plot_projection(ax_xy, points, (0, 1), "XY Projection")
    _plot_projection(ax_xz, points, (0, 2), "XZ Projection")
    _plot_3d(ax_3d, points)

    step_idx = np.arange(1, len(stats["step_norm"]) + 1)
    ax_step.plot(step_idx, stats["step_norm"], marker="o", linewidth=1.4, label="step length")
    ax_step.plot(np.arange(1, len(stats["cumulative"]) + 1), stats["cumulative"], linewidth=1.2, label="cumulative length")
    ax_step.set_title("Sequential Generation")
    ax_step.set_xlabel("point index")
    ax_step.grid(alpha=0.25)
    ax_step.legend(fontsize=8)

    accel_idx = np.arange(2, 2 + len(stats["accel_norm"]))
    turn_idx = np.arange(2, 2 + len(stats["turn_cos"]))
    if len(stats["accel_norm"]):
        ax_smooth.plot(accel_idx, stats["accel_norm"], marker="o", linewidth=1.4, label="2nd diff norm")
    if len(stats["turn_cos"]):
        ax_smooth.plot(turn_idx, stats["turn_cos"], marker="s", linewidth=1.2, label="turn cosine")
    ax_smooth.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax_smooth.set_title("Smoothness Constraint")
    ax_smooth.set_xlabel("point index")
    ax_smooth.grid(alpha=0.25)
    ax_smooth.legend(fontsize=8)

    if curvature is not None:
        ax_curv.plot(np.arange(1, len(curvature) + 1), curvature, linewidth=1.4, label="curvature")
    if torsion is not None:
        ax_curv.plot(np.arange(1, len(torsion) + 1), torsion, linewidth=1.2, label="torsion")
    ax_curv.set_title("Per-point Geometry")
    ax_curv.set_xlabel("point index")
    ax_curv.grid(alpha=0.25)
    ax_curv.legend(fontsize=8)

    summary = (
        f"{case_id} | {prefix}\n"
        f"risk={meta.get('risk', np.nan):.3f}, event={int(meta.get('event', 0))}, time={meta.get('event_time', np.nan):.1f}\n"
        f"path_length={stats['path_length']:.2f}, endpoint_disp={stats['endpoint_disp']:.2f}\n"
        f"step_mean={stats['step_mean']:.2f}, step_std={stats['step_std']:.2f}\n"
        f"accel_mean={stats['accel_mean']:.2f}, accel_max={stats['accel_max']:.2f}, turn_cos_mean={stats['turn_cos_mean']:.3f}"
    )
    fig.suptitle("Curve32 Geometry Constraint Visualization", fontsize=15)
    fig.text(0.50, 0.02, summary, ha="center", va="bottom", fontsize=10, family="monospace")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 32-point geometry constraints: ordered generation and smoothness.")
    parser.add_argument("--tensor_path", required=True, help="Path to exported *_tensors.pt")
    parser.add_argument("--case_csv", required=True, help="Path to exported *_cases.csv")
    parser.add_argument("--case_ids", default=None, help="Comma-separated case IDs")
    parser.add_argument("--case_table", default=None, help="CSV with case_id column")
    parser.add_argument("--top_n", type=int, default=4)
    parser.add_argument("--prefix", required=True, choices=["wsi_curve", "rna_curve"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    tensors = torch.load(args.tensor_path, map_location="cpu")
    case_df = pd.read_csv(args.case_csv)
    case_ids = _resolve_case_ids(args)
    all_case_ids = [str(x) for x in tensors["case_id"]]

    points_key = f"{args.prefix}_points"
    curv_key = f"{args.prefix}_curvature_per_point"
    tors_key = f"{args.prefix}_torsion_per_point"
    if points_key not in tensors:
        raise KeyError(f"{points_key} not found in {args.tensor_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        idx = _case_index(all_case_ids, case_id)
        if idx is None:
            continue
        row = case_df.loc[case_df["case_id"] == case_id]
        if row.empty:
            continue
        meta = row.iloc[0].to_dict()
        points = tensors[points_key][idx].cpu().numpy()
        curvature = tensors[curv_key][idx].cpu().numpy() if curv_key in tensors else None
        torsion = tensors[tors_key][idx].cpu().numpy() if tors_key in tensors else None
        out_path = out_dir / f"{case_id}_{args.prefix}_constraints.png"
        _plot_curve_constraints(
            case_id=case_id,
            prefix=args.prefix,
            points=points,
            curvature=curvature,
            torsion=torsion,
            meta=meta,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
