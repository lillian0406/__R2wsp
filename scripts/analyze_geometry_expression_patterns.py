from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _label_from_net(x: float) -> str:
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def _load_diag(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no diagnostic CSV matched: {pattern}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize geometry-feature expression patterns for positive vs negative rescue cases.")
    parser.add_argument("--rescue_csv", required=True)
    parser.add_argument("--diag_glob", required=True)
    parser.add_argument("--prefix", required=True, choices=["wsi_curve", "rna_curve"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rescue_df = pd.read_csv(args.rescue_csv)
    diag_df = _load_diag(args.diag_glob)
    rescue_df["gain_sign"] = rescue_df["net_rescue_pairs"].map(_label_from_net)

    keep_cols = [
        "fold",
        "seed",
        "case_id",
        "gain_sign",
        "net_rescue_pairs",
        "rescued_pairs",
        "harmed_pairs",
        "risk_bucket",
        "event",
        "event_time",
    ]
    prefix_cols = [c for c in diag_df.columns if c.startswith(f"{args.prefix}_")]
    common_cols = [
        "risk",
        "wsi_case_norm",
        "rna_case_norm",
        "fused_norm",
        "wsi_norm_share",
        "rna_norm_share",
        "gate_mean",
        "cross_modal_interaction_norm",
    ]
    use_cols = [c for c in common_cols + prefix_cols if c in diag_df.columns]
    merged = rescue_df[keep_cols].merge(diag_df[["fold", "seed", "case_id"] + use_cols], on=["fold", "seed", "case_id"], how="inner")
    if merged.empty:
        raise RuntimeError("merged rescue and diagnostics dataframe is empty")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "merged_case_expression.csv", index=False)

    stat_rows: list[dict[str, object]] = []
    sign_order = ["positive", "zero", "negative"]
    numeric_cols = [c for c in use_cols if np.issubdtype(merged[c].dtype, np.number)]
    for col in numeric_cols:
        row: dict[str, object] = {"feature": col}
        for sign in sign_order:
            subset = merged.loc[merged["gain_sign"] == sign, col].dropna().to_numpy(dtype=float)
            row[f"{sign}_n"] = int(len(subset))
            row[f"{sign}_mean"] = None if len(subset) == 0 else float(subset.mean())
            row[f"{sign}_std"] = None if len(subset) == 0 else float(subset.std())
        pos = merged.loc[merged["gain_sign"] == "positive", col].dropna().to_numpy(dtype=float)
        neg = merged.loc[merged["gain_sign"] == "negative", col].dropna().to_numpy(dtype=float)
        if len(pos) and len(neg):
            row["pos_minus_neg"] = float(pos.mean() - neg.mean())
            pooled = np.sqrt((pos.var() + neg.var()) / 2.0 + 1e-8)
            row["effect_size_like"] = float((pos.mean() - neg.mean()) / pooled)
        else:
            row["pos_minus_neg"] = None
            row["effect_size_like"] = None
        stat_rows.append(row)

    stat_rows.sort(
        key=lambda x: abs(float(x["effect_size_like"])) if x["effect_size_like"] is not None else -1.0,
        reverse=True,
    )
    _write_csv(
        out_dir / "expression_feature_summary.csv",
        stat_rows,
        [
            "feature",
            "positive_n",
            "positive_mean",
            "positive_std",
            "zero_n",
            "zero_mean",
            "zero_std",
            "negative_n",
            "negative_mean",
            "negative_std",
            "pos_minus_neg",
            "effect_size_like",
        ],
    )

    bucket_rows: list[dict[str, object]] = []
    bucket_table = (
        merged.groupby(["gain_sign", "risk_bucket"]).size().reset_index(name="count").sort_values(["gain_sign", "risk_bucket"])
    )
    bucket_rows.extend(bucket_table.to_dict(orient="records"))
    _write_csv(out_dir / "expression_risk_bucket_counts.csv", bucket_rows, ["gain_sign", "risk_bucket", "count"])

    top_pos = merged.sort_values(["net_rescue_pairs", "rescued_pairs"], ascending=False).head(20)
    top_neg = merged.sort_values(["net_rescue_pairs", "harmed_pairs"], ascending=[True, False]).head(20)
    top_pos.to_csv(out_dir / "top_positive_cases.csv", index=False)
    top_neg.to_csv(out_dir / "top_negative_cases.csv", index=False)

    md_lines = [
        f"# Expression Pattern Summary: {args.prefix}",
        "",
        "## Top Features By Positive-vs-Negative Separation",
        "",
        "| feature | positive_mean | negative_mean | delta | effect_size_like |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in stat_rows[:15]:
        md_lines.append(
            "| {feature} | {pm} | {nm} | {delta} | {eff} |".format(
                feature=row["feature"],
                pm="-" if row["positive_mean"] is None else f"{float(row['positive_mean']):.4f}",
                nm="-" if row["negative_mean"] is None else f"{float(row['negative_mean']):.4f}",
                delta="-" if row["pos_minus_neg"] is None else f"{float(row['pos_minus_neg']):+.4f}",
                eff="-" if row["effect_size_like"] is None else f"{float(row['effect_size_like']):+.4f}",
            )
        )
    (out_dir / "expression_feature_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "prefix": args.prefix,
        "n_cases": int(len(merged)),
        "gain_sign_counts": merged["gain_sign"].value_counts().to_dict(),
        "top_features": [row["feature"] for row in stat_rows[:10]],
    }
    (out_dir / "expression_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
