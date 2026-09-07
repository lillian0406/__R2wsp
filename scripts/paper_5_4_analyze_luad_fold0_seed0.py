#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pair_sign_matrix(risk: np.ndarray, times: np.ndarray) -> np.ndarray:
    rr = risk.reshape(-1, 1)
    tt = times.reshape(-1, 1)
    dr = rr - rr.T
    dt = tt - tt.T
    m = np.sign(dr * (-dt))
    np.fill_diagonal(m, 0.0)
    return m.astype(np.int8)


def _flip_stats(S_base: np.ndarray, S_other: np.ndarray, times: np.ndarray) -> tuple[int, int, float, np.ndarray]:
    N = int(times.shape[0])
    tri = np.triu_indices(N, k=1)
    tb = times[tri[0]]
    to = times[tri[1]]
    comparable = tb != to
    Sb = S_base[tri]
    So = S_other[tri]
    comparable = comparable & (Sb != 0) & (So != 0)
    flip = comparable & (Sb != So)
    valid = int(comparable.sum())
    flips = int(flip.sum())
    rate = float(flips / max(1, valid))
    D = np.zeros_like(S_base, dtype=np.uint8)
    D[tri] = flip.astype(np.uint8)
    D[(tri[1], tri[0])] = flip.astype(np.uint8)
    return valid, flips, rate, D


def _load_test_risk(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = _read_csv_rows(path)
    case_ids = [str(r["case_id"]) for r in rows]
    risk = np.asarray([float(r["risk"]) for r in rows], dtype=np.float64)
    times = np.asarray([float(r["event_time"]) for r in rows], dtype=np.float64)
    return case_ids, risk, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--cohort", default="LUAD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--export_tag", default="paper_5_4_final")
    args = ap.parse_args()

    out_base = Path(args.out_base).resolve()
    cohort = str(args.cohort)
    seed = int(args.seed)
    fold = int(args.fold)
    export_tag = str(args.export_tag)

    methods = ["fixed_off", "param_safe"]
    variants = ["u_only", "all_correct", "all_masked"]

    def export_csv(method: str, variant: str) -> Path:
        run_dir = out_base / f"{method}_{variant}" / cohort / f"seed{seed}" / f"fold{fold}" / "stage2"
        return run_dir / "case_risk_exports" / export_tag / "test_cases.csv"

    ref_case_ids: list[str] | None = None
    ref_times: np.ndarray | None = None
    S: dict[tuple[str, str], np.ndarray] = {}

    for method in methods:
        for variant in variants:
            p = export_csv(method, variant)
            if not p.exists():
                raise FileNotFoundError(str(p))
            case_ids, risk, times = _load_test_risk(p)
            if ref_case_ids is None:
                ref_case_ids = case_ids
                ref_times = times
            else:
                if case_ids != ref_case_ids:
                    raise ValueError(f"case_id mismatch for {method}/{variant}")
                if not np.allclose(times, ref_times, atol=0.0, rtol=0.0):
                    raise ValueError(f"event_time mismatch for {method}/{variant}")
            S[(method, variant)] = _pair_sign_matrix(risk, times)

    assert ref_case_ids is not None
    assert ref_times is not None

    rows_out: list[dict[str, object]] = []
    heatmaps: dict[tuple[str, str], np.ndarray] = {}
    for method in methods:
        base = S[(method, "u_only")]
        for variant in ["all_correct", "all_masked"]:
            valid, flips, rate, D = _flip_stats(base, S[(method, variant)], ref_times)
            rows_out.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                    "variant": variant,
                    "n_u": len(ref_case_ids),
                    "valid_pairs": valid,
                    "flip_pairs": flips,
                    "flip_rate": rate,
                }
            )
            heatmaps[(method, variant)] = D

    out_dir = out_base / "_analysis"
    _write_csv(
        out_dir / "flip_rate.csv",
        rows_out,
        ["cohort", "seed", "fold", "method", "variant", "n_u", "valid_pairs", "flip_pairs", "flip_rate"],
    )
    (out_dir / "manifest.json").write_text(json.dumps({"rows": rows_out}, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = [f"{r['method']}\n{r['variant']}" for r in rows_out]
    values = [float(r["flip_rate"]) for r in rows_out]
    plt.figure(figsize=(7.5, 3.5), dpi=160)
    x = np.arange(len(values))
    plt.bar(x, values, color=["#4C78A8", "#4C78A8", "#F58518", "#F58518"])
    plt.xticks(x, labels, fontsize=9)
    plt.ylabel("Flip rate (S_u vs S_x)")
    plt.title(f"{cohort} censoring-induced pairwise rank flips (U=test, N={len(ref_case_ids)})")
    plt.ylim(0.0, max(values) * 1.2 + 1e-6)
    plt.tight_layout()
    plt.savefig(out_dir / "flip_rate_bar.png")
    plt.close()

    for (method, variant), D in heatmaps.items():
        plt.figure(figsize=(4.2, 4.2), dpi=160)
        plt.imshow(D, cmap="magma", interpolation="nearest")
        plt.title(f"{cohort} ΔS heatmap | {method} | {variant}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"deltaS_{method}_{variant}.png")
        plt.close()


if __name__ == "__main__":
    main()

