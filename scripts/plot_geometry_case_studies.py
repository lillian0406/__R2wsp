from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def _resolve_case_ids(args: argparse.Namespace) -> list[str]:
    if args.case_ids:
        return [x.strip() for x in args.case_ids.split(",") if x.strip()]
    if not args.case_table:
        raise ValueError("either --case_ids or --case_table must be provided")
    df = pd.read_csv(args.case_table)
    return [str(x) for x in df["case_id"].head(int(args.top_n)).tolist()]


def _case_index(case_ids: list[str], target: str) -> int | None:
    for idx, case_id in enumerate(case_ids):
        if str(case_id) == str(target):
            return idx
    return None


def _plot_curve(ax, points, title: str, x_idx: int, y_idx: int) -> None:
    ax.plot(points[:, x_idx], points[:, y_idx], marker="o", linewidth=1.5, markersize=3)
    for step, (x_val, y_val) in enumerate(zip(points[:, x_idx], points[:, y_idx])):
        ax.text(float(x_val), float(y_val), str(step), fontsize=6)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latent geometry case studies from exported case_diagnostics tensors.")
    parser.add_argument("--tensor_path", required=True, help="Path to test_tensors.pt")
    parser.add_argument("--case_csv", required=True, help="Path to test_cases.csv")
    parser.add_argument("--case_table", default=None, help="Optional rescue_case_ranking.csv or similar with case_id column")
    parser.add_argument("--case_ids", default=None, help="Comma-separated explicit case IDs")
    parser.add_argument("--top_n", type=int, default=4)
    parser.add_argument("--prefix", default="rna_curve", choices=["rna_curve", "wsi_curve"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    case_ids = _resolve_case_ids(args)
    tensors = torch.load(args.tensor_path, map_location="cpu")
    diag_df = pd.read_csv(args.case_csv)
    all_case_ids = [str(x) for x in tensors["case_id"]]
    points_key = f"{args.prefix}_points"
    curv_key = f"{args.prefix}_curvature_per_point"
    tors_key = f"{args.prefix}_torsion_per_point"
    if points_key not in tensors:
        raise KeyError(f"{points_key} not found in tensor export")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        idx = _case_index(all_case_ids, case_id)
        if idx is None:
            continue
        row = diag_df.loc[diag_df["case_id"] == case_id].iloc[0]
        points = tensors[points_key][idx].numpy()
        curvature = tensors[curv_key][idx].numpy() if curv_key in tensors else None
        torsion = tensors[tors_key][idx].numpy() if tors_key in tensors else None

        fig = plt.figure(figsize=(12, 8))
        axes = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4),
        ]
        _plot_curve(axes[0], points, f"{args.prefix} XY", 0, 1)
        _plot_curve(axes[1], points, f"{args.prefix} XZ", 0, 2)
        _plot_curve(axes[2], points, f"{args.prefix} YZ", 1, 2)
        if curvature is not None:
            axes[3].plot(curvature, label="curvature", linewidth=1.5)
        if torsion is not None:
            axes[3].plot(torsion, label="torsion", linewidth=1.5)
        axes[3].set_title("Per-point Geometry")
        axes[3].grid(alpha=0.3)
        if curvature is not None or torsion is not None:
            axes[3].legend()

        fig.suptitle(
            f"{case_id} | risk={float(row['risk']):.3f} | event={int(row['event'])} | time={float(row['event_time']):.1f}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{case_id}_{args.prefix}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
