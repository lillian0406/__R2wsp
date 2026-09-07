#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics as st
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunRow:
    cohort: str
    A: float
    seed: int
    fold: int
    best_epoch: int | None
    best_val_c: float | None
    best_test_c: float | None
    final_test_c: float | None
    n_train_cases: int | None
    n_val_cases: int | None
    n_test_cases: int | None
    min_w: float | None
    mean_w: float | None
    frac_w_lt_05: float | None
    frac_w_lt_01: float | None
    source_dir: str


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return float(st.mean(xs)), float(st.pstdev(xs))


def _load_activation_stats(run_dir: Path) -> tuple[float | None, float | None, float | None, float | None]:
    hist = run_dir / "histories"
    if not hist.exists():
        return None, None, None, None
    csvs = sorted(hist.glob("seed*.csv"))
    if not csvs:
        return None, None, None, None
    p = csvs[0]
    w_list: list[float] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = _safe_float(row.get("window_weight_now"))
            if w is None:
                continue
            w_list.append(w)
    if not w_list:
        return None, None, None, None
    min_w = float(min(w_list))
    mean_w = float(st.mean(w_list))
    frac_w_lt_05 = float(sum(1 for w in w_list if w < 0.5) / len(w_list))
    frac_w_lt_01 = float(sum(1 for w in w_list if w < 0.1) / len(w_list))
    return min_w, mean_w, frac_w_lt_05, frac_w_lt_01


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_root = project_root / "outputs" / "k3_grid_A_param_safe_bs16_ep50_seed0"
    save_dir = out_root / "_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[RunRow] = []
    for sum_path in sorted(out_root.glob("A*/**/stage2/summary.json")):
        run_dir = sum_path.parent
        try:
            d = _read_json(sum_path)
        except Exception:
            continue
        parts = sum_path.parts
        try:
            a_part = next(x for x in parts if x.startswith("A"))
            A = float(a_part[1:])
        except Exception:
            continue
        try:
            cohort = parts[parts.index(a_part) + 1]
        except Exception:
            cohort = "UNK"
        seed = 0
        fold = 0
        for x in parts:
            if x.startswith("seed"):
                seed = int(x.replace("seed", ""))
            if x.startswith("fold"):
                fold = int(x.replace("fold", ""))

        min_w, mean_w, frac05, frac01 = _load_activation_stats(run_dir)
        run_rows.append(
            RunRow(
                cohort=str(cohort),
                A=float(A),
                seed=int(seed),
                fold=int(fold),
                best_epoch=_safe_int(d.get("best_epoch")),
                best_val_c=_safe_float(d.get("best_val_c_index")),
                best_test_c=_safe_float(d.get("best_test_c_index")),
                final_test_c=_safe_float(d.get("final_test_c_index")),
                n_train_cases=_safe_int(d.get("n_train_cases")),
                n_val_cases=_safe_int(d.get("n_val_cases")),
                n_test_cases=_safe_int(d.get("n_test_cases")),
                min_w=min_w,
                mean_w=mean_w,
                frac_w_lt_05=frac05,
                frac_w_lt_01=frac01,
                source_dir=str(run_dir),
            )
        )

    run_csv = save_dir / "runs.csv"
    with run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(run_rows[0]).keys()) if run_rows else [])
        writer.writeheader()
        for r in run_rows:
            writer.writerow(asdict(r))

    agg: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for r in run_rows:
        key = (r.cohort, f"{r.A:.3f}")
        agg.setdefault(r.cohort, {}).setdefault(f"{r.A:.3f}", {})

    for cohort in sorted({r.cohort for r in run_rows}):
        for a_str in sorted({f"{r.A:.3f}" for r in run_rows if r.cohort == cohort}):
            subset = [r for r in run_rows if r.cohort == cohort and f"{r.A:.3f}" == a_str]
            best_tests = [r.best_test_c for r in subset if r.best_test_c is not None]
            final_tests = [r.final_test_c for r in subset if r.final_test_c is not None]
            min_ws = [r.min_w for r in subset if r.min_w is not None]
            mean_ws = [r.mean_w for r in subset if r.mean_w is not None]
            frac01s = [r.frac_w_lt_01 for r in subset if r.frac_w_lt_01 is not None]
            m_bt, s_bt = _mean_std(best_tests)
            m_ft, s_ft = _mean_std(final_tests)
            m_minw, s_minw = _mean_std(min_ws)
            m_meanw, s_meanw = _mean_std(mean_ws)
            m_f01, s_f01 = _mean_std(frac01s)
            agg[cohort][a_str] = {
                "n": int(len(subset)),
                "best_test_mean": m_bt,
                "best_test_std": s_bt,
                "final_test_mean": m_ft,
                "final_test_std": s_ft,
                "min_w_mean": m_minw,
                "min_w_std": s_minw,
                "mean_w_mean": m_meanw,
                "mean_w_std": s_meanw,
                "frac_w_lt_01_mean": m_f01,
                "frac_w_lt_01_std": s_f01,
            }

    agg_json = save_dir / "aggregate.json"
    agg_json.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    table_csv = save_dir / "aggregate_table.csv"
    with table_csv.open("w", newline="") as f:
        fieldnames = [
            "cohort",
            "A",
            "n",
            "best_test_mean",
            "best_test_std",
            "final_test_mean",
            "final_test_std",
            "min_w_mean",
            "mean_w_mean",
            "frac_w_lt_01_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cohort, by_a in agg.items():
            for a_str, s in by_a.items():
                writer.writerow(
                    {
                        "cohort": cohort,
                        "A": a_str,
                        "n": s.get("n"),
                        "best_test_mean": s.get("best_test_mean"),
                        "best_test_std": s.get("best_test_std"),
                        "final_test_mean": s.get("final_test_mean"),
                        "final_test_std": s.get("final_test_std"),
                        "min_w_mean": s.get("min_w_mean"),
                        "mean_w_mean": s.get("mean_w_mean"),
                        "frac_w_lt_01_mean": s.get("frac_w_lt_01_mean"),
                    }
                )


if __name__ == "__main__":
    main()

