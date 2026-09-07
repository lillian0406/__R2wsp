from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _match_fold_seed(paths: list[str]) -> dict[tuple[int, int], Path]:
    matched: dict[tuple[int, int], Path] = {}
    for path_str in sorted(paths):
        path = Path(path_str)
        m = re.search(r"_k(\d+)_seed(\d+)", str(path))
        if m is None:
            continue
        matched[(int(m.group(1)), int(m.group(2)))] = path
    return matched


def _load_tensor_table(path: Path) -> pd.DataFrame:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    case_ids = [str(x) for x in payload["case_id"]]
    rows: list[dict[str, object]] = []
    for idx, case_id in enumerate(case_ids):
        row: dict[str, object] = {"case_id": case_id}
        for key in ("wsi_case", "rna_case", "fused"):
            if key not in payload:
                continue
            vec = payload[key][idx]
            if hasattr(vec, "detach"):
                vec = vec.detach().cpu().numpy()
            vec = np.asarray(vec, dtype=np.float64).reshape(-1)
            row[key] = vec
        rows.append(row)
    return pd.DataFrame(rows)


def _explode_vector_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    vectors = np.stack(df[col].to_list(), axis=0)
    out = pd.DataFrame(vectors, columns=[f"{prefix}_{i}" for i in range(vectors.shape[1])])
    return pd.concat([df[["fold", "seed", "case_id", "gain_sign"]].reset_index(drop=True), out], axis=1)


def _plot_spectrum(summary_df: pd.DataFrame, title: str, out_path: Path) -> None:
    x = summary_df["dim"].to_numpy(dtype=int)
    pos = summary_df["positive_mean_delta"].to_numpy(dtype=float)
    neg = summary_df["negative_mean_delta"].to_numpy(dtype=float)
    allv = summary_df["all_mean_delta"].to_numpy(dtype=float)

    plt.figure(figsize=(12, 4.8))
    plt.plot(x, allv, linewidth=1.7, label="all cases")
    plt.plot(x, pos, linewidth=1.7, label="positive rescue cases")
    plt.plot(x, neg, linewidth=1.7, label="negative rescue cases")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Latent dimension")
    plt.ylabel("Geo - Baseline mean delta")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_markdown(results: dict[str, pd.DataFrame], out_path: Path) -> None:
    lines = ["# Geo vs Baseline Latent Spectra", ""]
    for name, df in results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| dim | positive_mean_delta | negative_mean_delta | pos_minus_neg | abs_all_mean_delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        top = df.reindex(df["pos_minus_neg"].abs().sort_values(ascending=False).index).head(15)
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['dim'])} | {row['positive_mean_delta']:+.4f} | {row['negative_mean_delta']:+.4f} | "
                f"{row['pos_minus_neg']:+.4f} | {abs(float(row['all_mean_delta'])):.4f} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare latent-dimension spectra between a no-geo baseline and a geometry run.")
    parser.add_argument("--baseline_glob", required=True, help="Glob for baseline test_tensors.pt files")
    parser.add_argument("--geometry_glob", required=True, help="Glob for geometry test_tensors.pt files")
    parser.add_argument("--rescue_csv", required=True, help="Rescue case ranking CSV used to label positive/negative geometry cases")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    baseline_map = _match_fold_seed(glob.glob(args.baseline_glob))
    geometry_map = _match_fold_seed(glob.glob(args.geometry_glob))
    shared = sorted(set(baseline_map) & set(geometry_map))
    if not shared:
        raise RuntimeError("no shared fold/seed test_tensors found")

    rescue_df = pd.read_csv(args.rescue_csv)
    rescue_df["gain_sign"] = rescue_df["net_rescue_pairs"].map(lambda x: "positive" if x > 0 else ("negative" if x < 0 else "zero"))
    rescue_df["case_id"] = rescue_df["case_id"].astype(str)

    merged_rows = []
    for fold, seed in shared:
        base_df = _load_tensor_table(baseline_map[(fold, seed)])
        geom_df = _load_tensor_table(geometry_map[(fold, seed)])
        base_df["fold"] = fold
        base_df["seed"] = seed
        geom_df["fold"] = fold
        geom_df["seed"] = seed
        pair = base_df.merge(geom_df, on=["fold", "seed", "case_id"], suffixes=("_base", "_geom"))
        pair = pair.merge(rescue_df[["fold", "seed", "case_id", "gain_sign"]], on=["fold", "seed", "case_id"], how="left")
        pair["gain_sign"] = pair["gain_sign"].fillna("zero")
        merged_rows.append(pair)
    merged = pd.concat(merged_rows, ignore_index=True)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame] = {}
    for rep in ("wsi_case", "rna_case", "fused"):
        if f"{rep}_base" not in merged.columns or f"{rep}_geom" not in merged.columns:
            continue
        work = merged[["fold", "seed", "case_id", "gain_sign", f"{rep}_base", f"{rep}_geom"]].copy()
        work[rep] = [np.asarray(g, dtype=np.float64) - np.asarray(b, dtype=np.float64) for b, g in zip(work[f"{rep}_base"], work[f"{rep}_geom"])]
        wide = _explode_vector_column(work[["fold", "seed", "case_id", "gain_sign", rep]].rename(columns={rep: "vec"}), "vec", rep)
        stat_rows: list[dict[str, float]] = []
        feature_cols = [c for c in wide.columns if c.startswith(f"{rep}_")]
        for idx, col in enumerate(feature_cols):
            all_vals = wide[col].to_numpy(dtype=float)
            pos_vals = wide.loc[wide["gain_sign"] == "positive", col].to_numpy(dtype=float)
            neg_vals = wide.loc[wide["gain_sign"] == "negative", col].to_numpy(dtype=float)
            stat_rows.append(
                {
                    "dim": idx,
                    "all_mean_delta": float(all_vals.mean()),
                    "positive_mean_delta": float(pos_vals.mean()) if len(pos_vals) else 0.0,
                    "negative_mean_delta": float(neg_vals.mean()) if len(neg_vals) else 0.0,
                    "pos_minus_neg": (float(pos_vals.mean()) if len(pos_vals) else 0.0) - (float(neg_vals.mean()) if len(neg_vals) else 0.0),
                }
            )
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(out_dir / f"{rep}_delta_spectrum.csv", index=False)
        _plot_spectrum(stat_df, f"{rep}: geometry vs no-geo baseline", out_dir / f"{rep}_delta_spectrum.png")
        summaries[rep] = stat_df

    _write_markdown(summaries, out_dir / "latent_spectra_summary.md")
    meta = {
        "shared_fold_seed": [list(x) for x in shared],
        "n_cases": int(len(merged)),
        "gain_sign_counts": merged["gain_sign"].value_counts().to_dict(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
