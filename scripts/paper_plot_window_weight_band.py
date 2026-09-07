#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _collect_histories(root: Path, cohort: str) -> list[Path]:
    paths: list[Path] = []
    for seed in [0, 1, 2]:
        for fold in [0, 1, 2, 3, 4]:
            p = root / cohort / f"seed{seed}" / f"fold{fold}" / "stage2" / "histories" / f"seed{seed}.csv"
            if p.exists():
                paths.append(p)
    return paths


def _epoch_series(path: Path, col: str, max_epoch: int | None) -> pd.Series:
    df = pd.read_csv(path)
    if "epoch" not in df.columns or col not in df.columns:
        raise ValueError(f"missing columns epoch/{col}: {path}")
    df = df[["epoch", col]].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="raise").astype(int)
    df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    if max_epoch is not None:
        df = df[df["epoch"] <= int(max_epoch)]
    df = df.sort_values("epoch")
    s = pd.Series(df[col].to_numpy(dtype=np.float64), index=df["epoch"].to_numpy(dtype=np.int64))
    return s


def _aggregate(paths: list[Path], col: str, max_epoch: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [_epoch_series(p, col, max_epoch) for p in paths]
    epochs = sorted(set().union(*[set(s.index.tolist()) for s in series]))
    mat = np.full((len(series), len(epochs)), np.nan, dtype=np.float64)
    for i, s in enumerate(series):
        for j, e in enumerate(epochs):
            if e in s.index:
                mat[i, j] = float(s.loc[e])
    mean = np.nanmean(mat, axis=0)
    std = np.sqrt(np.nanmean((mat - mean[None, :]) ** 2, axis=0))
    n = np.sum(np.isfinite(mat), axis=0)
    return np.asarray(epochs, dtype=np.int64), mean, std, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="LUAD")
    ap.add_argument("--reg_root", default="outputs/twostage_param_safe_bs16_V1")
    ap.add_argument("--ctrl_root", default="outputs/twostage_fixed_off_bs16_V1")
    ap.add_argument("--col", default="window_weight_now")
    ap.add_argument("--max_epoch", type=int, default=None)
    ap.add_argument("--out_dir", default="outputs/figures_paper")
    args = ap.parse_args()

    cohort = str(args.cohort).upper()
    reg_root = Path(args.reg_root).resolve()
    ctrl_root = Path(args.ctrl_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_paths = _collect_histories(reg_root, cohort)
    ctrl_paths = _collect_histories(ctrl_root, cohort)
    if len(reg_paths) == 0 or len(ctrl_paths) == 0:
        raise SystemExit(f"missing histories: reg={len(reg_paths)} ctrl={len(ctrl_paths)}")

    epochs_r, mean_r, std_r, n_r = _aggregate(reg_paths, args.col, args.max_epoch)
    epochs_c, mean_c, std_c, n_c = _aggregate(ctrl_paths, args.col, args.max_epoch)

    max_epoch = int(max(epochs_r.max(), epochs_c.max()))
    fig = plt.figure(figsize=(9.2, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"{cohort} — w_now across epochs (mean±std over 15 runs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("w_now")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(1, max_epoch)
    ax.grid(True, alpha=0.25)

    ax.plot(epochs_r, mean_r, color="#d95f02", linewidth=2.0, label="param_safe (reg)")
    ax.fill_between(epochs_r, mean_r - std_r, mean_r + std_r, color="#d95f02", alpha=0.18)
    ax.plot(epochs_c, mean_c, color="#1f78b4", linewidth=2.0, linestyle="--", label="fixed_off (ctrl)")
    ax.fill_between(epochs_c, mean_c - std_c, mean_c + std_c, color="#1f78b4", alpha=0.12)

    ax.legend(frameon=False, loc="upper right")

    out_png = out_dir / f"{cohort}_w_now_band_15runs.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    out_json = out_dir / f"{cohort}_w_now_band_15runs.json"
    out_json.write_text(
        json.dumps(
            {
                "cohort": cohort,
                "col": args.col,
                "reg_root": str(reg_root),
                "ctrl_root": str(ctrl_root),
                "n_reg_paths": len(reg_paths),
                "n_ctrl_paths": len(ctrl_paths),
                "out_png": str(out_png),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

