#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    den = float(np.sqrt((xr * xr).sum()) * np.sqrt((yr * yr).sum()))
    if den <= 0:
        return float("nan")
    return float((xr * yr).sum() / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_csv", required=True, help="P1(best) exported test_cases.csv")
    ap.add_argument("--p2_best_csv", required=True, help="P2(best) exported test_cases.csv")
    ap.add_argument("--p2_final_csv", required=True, help="P2(final) exported test_cases.csv")
    ap.add_argument("--out", required=True, help="Output image path (.png/.pdf)")
    ap.add_argument("--title", default="Risk change across stages (fold0 seed0)")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    p1 = pd.read_csv(Path(args.p1_csv))
    p2b = pd.read_csv(Path(args.p2_best_csv))
    p2f = pd.read_csv(Path(args.p2_final_csv))

    for df in (p1, p2b, p2f):
        if "case_id" not in df.columns or "risk" not in df.columns:
            raise ValueError("CSV must contain columns: case_id, risk")
        df["case_id"] = df["case_id"].astype(str)

    df = (
        p1[["case_id", "risk"]]
        .rename(columns={"risk": "P1(best)"})
        .merge(p2b[["case_id", "risk"]].rename(columns={"risk": "P2(best)"}), on="case_id", how="inner")
        .merge(p2f[["case_id", "risk"]].rename(columns={"risk": "P2(final)"}), on="case_id", how="inner")
    )
    if df.empty:
        raise ValueError("No overlapping case_id across the three CSV files.")

    rho = _spearman(df["P1(best)"].to_numpy(), df["P2(final)"].to_numpy())
    df["delta_abs"] = (df["P2(final)"] - df["P1(best)"]).abs()
    topk = int(args.topk)
    top = df.sort_values("delta_abs", ascending=False).head(topk).copy()
    mean_abs_delta = float(top["delta_abs"].mean()) if not top.empty else float("nan")

    stages = ["P1(best)", "P2(best)", "P2(final)"]
    stage_data = [df[s].to_numpy(dtype=np.float64) for s in stages]
    fig = plt.figure(figsize=(12.2, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    parts = ax0.violinplot(stage_data, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c72b0")
        pc.set_edgecolor("#333333")
        pc.set_alpha(0.35)
    ax0.boxplot(stage_data, widths=0.18, patch_artist=True, showfliers=False, boxprops=dict(facecolor="white", alpha=0.8))
    ax0.set_title("Risk distribution (Phase-2 test)")
    ax0.set_xlabel("")
    ax0.set_xticks([1, 2, 3], stages)
    ax0.set_ylabel("Predicted risk")

    ax1 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(3, dtype=np.float64)
    for _, r in top.iterrows():
        y = np.asarray([r["P1(best)"], r["P2(best)"], r["P2(final)"]], dtype=np.float64)
        ax1.plot(x_pos, y, color="#7b3294", alpha=0.55, linewidth=1.2)
        ax1.scatter(x_pos, y, color="#7b3294", alpha=0.85, s=16)
    ax1.set_xticks([0, 1, 2], ["P1(best)", "P2(best)", "P2(final)"])
    ax1.set_ylabel("Predicted risk")
    ax1.set_title(f"Top-{topk} |ΔRisk| cases (slope graph)")
    ax1.text(
        0.02,
        0.98,
        f"Spearman(P1,P2final)={rho:.3f}\nMean|ΔS|={mean_abs_delta:.4f}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
    )

    fig.suptitle(str(args.title), y=1.02, fontsize=13)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
