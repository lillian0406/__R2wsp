#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_hist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"missing epoch column: {path}")
    return df


def _slice_epochs(df: pd.DataFrame, *, max_epoch: int) -> pd.DataFrame:
    df = df[df["epoch"].astype(int) <= int(max_epoch)].copy()
    df = df.sort_values("epoch")
    return df


def _get_col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise ValueError(f"missing column {name}")
    return df[name].to_numpy(dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg_hist", required=True)
    ap.add_argument("--ctrl_hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Pullback dynamics (fold0 seed0)")
    ap.add_argument("--max_epoch", type=int, default=8)
    ap.add_argument("--reg_label", default="Dynamic pullback (window)")
    ap.add_argument("--ctrl_label", default="No pullback effective (fixed_off)")
    args = ap.parse_args()

    reg = _slice_epochs(_load_hist(Path(args.reg_hist)), max_epoch=args.max_epoch)
    ctrl = _slice_epochs(_load_hist(Path(args.ctrl_hist)), max_epoch=args.max_epoch)

    x_reg = _get_col(reg, "epoch")
    x_ctrl = _get_col(ctrl, "epoch")

    w_reg = _get_col(reg, "window_weight_now")
    w_ctrl = _get_col(ctrl, "window_weight_now")

    l2_reg = _get_col(reg, "pullback_l2")
    l2_ctrl = _get_col(ctrl, "pullback_l2")

    pl_reg = _get_col(reg, "pullback_loss")
    pl_ctrl = _get_col(ctrl, "pullback_loss")

    loss_reg = _get_col(reg, "loss")
    loss_ctrl = _get_col(ctrl, "loss")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.4))
    fig.suptitle(str(args.title), y=1.02, fontsize=12.5)

    def draw(ax, y_reg, y_ctrl, ylabel, title):
        ax.plot(x_reg, y_reg, color="#d95f02", linewidth=2.0, label=args.reg_label)
        ax.plot(x_ctrl, y_ctrl, color="#1f78b4", linewidth=1.8, linestyle="--", label=args.ctrl_label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, int(args.max_epoch))
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x_reg, w_reg, color="#fdbf6f", linewidth=1.2, alpha=0.95, label="w_now (reg)")
        ax2.plot(x_ctrl, w_ctrl, color="#a6cee3", linewidth=1.2, alpha=0.95, label="w_now (ctrl)")
        ax2.set_ylabel("w_now")
        ax2.set_ylim(0.0, 1.05)
        return ax2

    ax2_0 = draw(axes[0], l2_reg, l2_ctrl, "pullback_l2", "pullback_l2")
    ax2_1 = draw(axes[1], pl_reg, pl_ctrl, "pullback_loss", "pullback_loss")
    ax2_2 = draw(axes[2], loss_reg, loss_ctrl, "total loss", "total loss")

    lines = []
    labels = []
    for ax in [axes[0], ax2_0]:
        h, l = ax.get_legend_handles_labels()
        lines += h
        labels += l
    fig.legend(lines, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

