#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _load_risk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_id" not in df.columns or "risk" not in df.columns:
        raise ValueError(f"risk csv must contain case_id,risk: {path}")
    df["case_id"] = df["case_id"].astype(str)
    df["risk"] = pd.to_numeric(df["risk"], errors="raise")
    return df[["case_id", "risk"]].copy()


def _jitter(n: int, *, scale: float = 0.06, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="LUAD")
    ap.add_argument("--stage1_root", default="outputs/twostage_pretrain_bs16_V1")
    ap.add_argument("--stage2_root", default="outputs/twostage_param_safe_bs16_V1")
    ap.add_argument("--out_dir", default="outputs/figures_paper")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    cohort = str(args.cohort).upper()
    stage1_root = Path(args.stage1_root).resolve()
    stage2_root = Path(args.stage2_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for seed in [0, 1, 2]:
        for fold in [0, 1, 2, 3, 4]:
            p1 = stage1_root / cohort / f"seed{seed}" / f"fold{fold}" / "stage1" / "case_risk_exports" / "best_phase2test" / "test_cases.csv"
            p2b = stage2_root / cohort / f"seed{seed}" / f"fold{fold}" / "stage2" / "case_risk_exports" / "best" / "test_cases.csv"
            p2f = stage2_root / cohort / f"seed{seed}" / f"fold{fold}" / "stage2" / "case_risk_exports" / "final" / "test_cases.csv"

            if not p1.exists() or not p2b.exists() or not p2f.exists():
                failures.append(
                    {
                        "cohort": cohort,
                        "seed": seed,
                        "fold": fold,
                        "missing_p1": str(p1) if not p1.exists() else "",
                        "missing_p2_best": str(p2b) if not p2b.exists() else "",
                        "missing_p2_final": str(p2f) if not p2f.exists() else "",
                    }
                )
                continue

            d1 = _load_risk_csv(p1).rename(columns={"risk": "p1"})
            d2b = _load_risk_csv(p2b).rename(columns={"risk": "p2b"})
            d2f = _load_risk_csv(p2f).rename(columns={"risk": "p2f"})
            df = d1.merge(d2f, on="case_id", how="inner").merge(d2b, on="case_id", how="inner")
            if df.empty:
                failures.append({"cohort": cohort, "seed": seed, "fold": fold, "error": "no overlapping case_id"})
                continue

            rho = _spearman(df["p1"].to_numpy(), df["p2f"].to_numpy())
            delta = (df["p2f"] - df["p1"]).to_numpy(dtype=np.float64)
            abs_delta = np.abs(delta)
            mean_abs_delta = float(abs_delta.mean())
            median_abs_delta = float(np.median(abs_delta))

            topk = int(args.topk)
            top = np.sort(abs_delta)[-topk:] if len(abs_delta) >= topk else abs_delta
            topk_mean_abs_delta = float(np.mean(top)) if len(top) > 0 else float("nan")

            rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "fold": fold,
                    "n_cases": int(len(df)),
                    "spearman_p1_p2final": float(rho),
                    "mean_abs_delta_p1_p2final": float(mean_abs_delta),
                    "median_abs_delta_p1_p2final": float(median_abs_delta),
                    "topk": topk,
                    "topk_mean_abs_delta_p1_p2final": float(topk_mean_abs_delta),
                }
            )

    out_csv = out_dir / f"{cohort}_risk_drift_15runs.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_fail = out_dir / f"{cohort}_risk_drift_15runs_missing.csv"
    pd.DataFrame(failures).to_csv(out_fail, index=False)

    if len(rows) == 0:
        raise SystemExit(f"no valid runs. missing list written to: {out_fail}")

    df = pd.DataFrame(rows)
    rho = df["spearman_p1_p2final"].to_numpy(dtype=np.float64)
    drift = df["mean_abs_delta_p1_p2final"].to_numpy(dtype=np.float64)

    def mean_std(x: np.ndarray) -> tuple[float, float]:
        m = float(np.mean(x))
        s = float(np.sqrt(np.mean((x - m) ** 2)))
        return m, s

    rho_m, rho_s = mean_std(rho)
    drift_m, drift_s = mean_std(drift)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.suptitle(f"{cohort} — Risk drift summary (15 runs = 3 seeds × 5 folds)", y=1.02, fontsize=12.5)

    axes[0].boxplot([rho], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#ccebc5", alpha=0.85))
    axes[0].scatter(np.ones_like(rho) + _jitter(len(rho), seed=0), rho, s=18, color="#2ca25f", alpha=0.75)
    axes[0].set_xticks([1], ["Spearman(P1,P2final)"])
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"mean±std = {rho_m:.3f}±{rho_s:.3f}")
    axes[0].grid(True, alpha=0.25)

    axes[1].boxplot([drift], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#fdd0a2", alpha=0.85))
    axes[1].scatter(np.ones_like(drift) + _jitter(len(drift), seed=1), drift, s=18, color="#e6550d", alpha=0.75)
    axes[1].set_xticks([1], ["Mean|ΔRisk|(P1→P2final)"])
    axes[1].set_ylabel("Mean absolute risk change")
    axes[1].set_title(f"mean±std = {drift_m:.3f}±{drift_s:.3f}")
    axes[1].grid(True, alpha=0.25)

    out_png = out_dir / f"{cohort}_risk_drift_15runs_boxplot.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    out_json = out_dir / f"{cohort}_risk_drift_15runs_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "cohort": cohort,
                "n_runs": int(len(df)),
                "spearman_p1_p2final_mean": rho_m,
                "spearman_p1_p2final_std_pop": rho_s,
                "mean_abs_delta_p1_p2final_mean": drift_m,
                "mean_abs_delta_p1_p2final_std_pop": drift_s,
                "out_csv": str(out_csv),
                "out_png": str(out_png),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
