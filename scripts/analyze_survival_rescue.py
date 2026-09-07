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


def _comparable_direction(time_i: float, event_i: float, time_j: float, event_j: float) -> tuple[int, int] | None:
    if event_i >= 0.5 and time_i < time_j:
        return 0, 1
    if event_j >= 0.5 and time_j < time_i:
        return 1, 0
    return None


def _quantile_bucket(values: np.ndarray, x: float) -> str:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    if x <= q1:
        return "low"
    if x <= q2:
        return "mid"
    return "high"


def _match_exports(pattern: str) -> dict[tuple[int, int], Path]:
    matched: dict[tuple[int, int], Path] = {}
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        df = pd.read_csv(path)
        if df.empty:
            continue
        fold = int(df["fold"].iloc[0])
        seed = int(df["seed"].iloc[0])
        matched[(fold, seed)] = path
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hard-pair rescue patterns between a baseline run family and a geometry-enhanced run family.")
    parser.add_argument("--baseline_glob", required=True, help="Glob for baseline exported case CSVs, e.g. /path/*/case_diagnostics/test_cases.csv")
    parser.add_argument("--geometry_glob", required=True, help="Glob for geometry exported case CSVs")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    baseline_map = _match_exports(args.baseline_glob)
    geometry_map = _match_exports(args.geometry_glob)
    shared_keys = sorted(set(baseline_map) & set(geometry_map))
    if not shared_keys:
        raise RuntimeError("no shared (fold, seed) exports found between baseline and geometry patterns")

    pair_rows: list[dict[str, object]] = []
    case_stats: dict[tuple[int, int, str], dict[str, object]] = {}
    run_rows: list[dict[str, object]] = []

    for fold, seed in shared_keys:
        base_df = pd.read_csv(baseline_map[(fold, seed)]).sort_values("case_id").reset_index(drop=True)
        geom_df = pd.read_csv(geometry_map[(fold, seed)]).sort_values("case_id").reset_index(drop=True)
        merged = base_df.merge(
            geom_df,
            on=["case_id", "split", "fold", "seed", "event_time", "event", "censorship"],
            suffixes=("_base", "_geom"),
        )
        if merged.empty:
            continue

        geom_risks = merged["risk_geom"].to_numpy(dtype=float)
        total_pairs = 0
        base_correct = 0
        geom_correct = 0
        rescue_pairs = 0
        harm_pairs = 0
        hard_baseline_errors = 0
        hard_rescues = 0
        hard_margins: list[float] = []
        pending_rows: list[dict[str, object]] = []

        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                direction = _comparable_direction(
                    float(merged.at[i, "event_time"]),
                    float(merged.at[i, "event"]),
                    float(merged.at[j, "event_time"]),
                    float(merged.at[j, "event"]),
                )
                if direction is None:
                    continue
                early, late = (i, j) if direction == (0, 1) else (j, i)
                total_pairs += 1
                base_margin = float(merged.at[early, "risk_base"] - merged.at[late, "risk_base"])
                geom_margin = float(merged.at[early, "risk_geom"] - merged.at[late, "risk_geom"])
                base_ok = base_margin > 0.0
                geom_ok = geom_margin > 0.0
                if base_ok:
                    base_correct += 1
                if geom_ok:
                    geom_correct += 1
                if (not base_ok) and geom_ok:
                    rescue_pairs += 1
                if base_ok and (not geom_ok):
                    harm_pairs += 1

                row = {
                    "fold": fold,
                    "seed": seed,
                    "early_case_id": merged.at[early, "case_id"],
                    "late_case_id": merged.at[late, "case_id"],
                    "early_time": float(merged.at[early, "event_time"]),
                    "late_time": float(merged.at[late, "event_time"]),
                    "base_margin": base_margin,
                    "geom_margin": geom_margin,
                    "base_correct": int(base_ok),
                    "geom_correct": int(geom_ok),
                    "rescued": int((not base_ok) and geom_ok),
                    "harmed": int(base_ok and (not geom_ok)),
                }
                pending_rows.append(row)
                hard_margins.append(abs(base_margin))

                for idx_case, role in ((early, "early"), (late, "late")):
                    key = (fold, seed, str(merged.at[idx_case, "case_id"]))
                    stats = case_stats.setdefault(
                        key,
                        {
                            "fold": fold,
                            "seed": seed,
                            "case_id": str(merged.at[idx_case, "case_id"]),
                            "event_time": float(merged.at[idx_case, "event_time"]),
                            "event": float(merged.at[idx_case, "event"]),
                            "risk_geom": float(merged.at[idx_case, "risk_geom"]),
                            "rescued_pairs": 0,
                            "harmed_pairs": 0,
                            "base_errors_involved": 0,
                            "pair_count": 0,
                            "role_early_count": 0,
                            "role_late_count": 0,
                        },
                    )
                    stats["pair_count"] = int(stats["pair_count"]) + 1
                    stats[f"role_{role}_count"] = int(stats[f"role_{role}_count"]) + 1
                    if not base_ok:
                        stats["base_errors_involved"] = int(stats["base_errors_involved"]) + 1
                    if (not base_ok) and geom_ok:
                        stats["rescued_pairs"] = int(stats["rescued_pairs"]) + 1
                    if base_ok and (not geom_ok):
                        stats["harmed_pairs"] = int(stats["harmed_pairs"]) + 1

        if pending_rows:
            margin_cut = float(np.quantile(np.asarray(hard_margins, dtype=np.float64), 0.25))
            for row in pending_rows:
                if abs(float(row["base_margin"])) <= margin_cut and not bool(row["base_correct"]):
                    hard_baseline_errors += 1
                    if bool(row["rescued"]):
                        hard_rescues += 1
            pair_rows.extend(pending_rows)

        run_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "n_cases": len(merged),
                "n_pairs": total_pairs,
                "base_pair_c_index": None if total_pairs == 0 else base_correct / total_pairs,
                "geom_pair_c_index": None if total_pairs == 0 else geom_correct / total_pairs,
                "rescue_pairs": rescue_pairs,
                "harm_pairs": harm_pairs,
                "net_rescue_pairs": rescue_pairs - harm_pairs,
                "rescue_rate_among_base_errors": None if (total_pairs - base_correct) == 0 else rescue_pairs / (total_pairs - base_correct),
                "hard_baseline_errors": hard_baseline_errors,
                "hard_rescues": hard_rescues,
                "hard_rescue_rate": None if hard_baseline_errors == 0 else hard_rescues / hard_baseline_errors,
            }
        )

    case_rows: list[dict[str, object]] = []
    if case_stats:
        all_geom_risks = np.asarray([float(x["risk_geom"]) for x in case_stats.values()], dtype=np.float64)
        for row in case_stats.values():
            row["net_rescue_pairs"] = int(row["rescued_pairs"]) - int(row["harmed_pairs"])
            row["risk_bucket"] = _quantile_bucket(all_geom_risks, float(row["risk_geom"]))
            case_rows.append(row)
        case_rows.sort(key=lambda x: (int(x["net_rescue_pairs"]), int(x["rescued_pairs"])), reverse=True)

    out_dir = Path(args.out_dir).resolve()
    _write_csv(
        out_dir / "rescue_pair_summary.csv",
        run_rows,
        [
            "fold",
            "seed",
            "n_cases",
            "n_pairs",
            "base_pair_c_index",
            "geom_pair_c_index",
            "rescue_pairs",
            "harm_pairs",
            "net_rescue_pairs",
            "rescue_rate_among_base_errors",
            "hard_baseline_errors",
            "hard_rescues",
            "hard_rescue_rate",
        ],
    )
    _write_csv(
        out_dir / "rescue_case_ranking.csv",
        case_rows,
        [
            "fold",
            "seed",
            "case_id",
            "event_time",
            "event",
            "risk_geom",
            "risk_bucket",
            "pair_count",
            "base_errors_involved",
            "rescued_pairs",
            "harmed_pairs",
            "net_rescue_pairs",
            "role_early_count",
            "role_late_count",
        ],
    )
    _write_csv(
        out_dir / "rescue_pair_detail.csv",
        pair_rows,
        [
            "fold",
            "seed",
            "early_case_id",
            "late_case_id",
            "early_time",
            "late_time",
            "base_margin",
            "geom_margin",
            "base_correct",
            "geom_correct",
            "rescued",
            "harmed",
        ],
    )

    summary = {
        "matched_runs": len(run_rows),
        "shared_fold_seed_keys": [f"{fold}_{seed}" for fold, seed in shared_keys],
        "total_rescue_pairs": int(sum(int(x["rescue_pairs"]) for x in run_rows)),
        "total_harm_pairs": int(sum(int(x["harm_pairs"]) for x in run_rows)),
        "total_net_rescue_pairs": int(sum(int(x["net_rescue_pairs"]) for x in run_rows)),
        "risk_bucket_counts": (
            pd.DataFrame(case_rows)["risk_bucket"].value_counts().to_dict() if case_rows else {}
        ),
    }
    (out_dir / "rescue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
