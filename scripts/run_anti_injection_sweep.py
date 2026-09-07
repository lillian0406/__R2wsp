#!/usr/bin/env python3
"""
Scheme A sweep: censored_stage_survival stage-1 (plip_luad_256 + rna_vec)
==========================================================================
Runs per fold 0..4 x seed 0..2 x [baseline, anti-injection] = 30 jobs,
SERIALLY on one GPU (CUDA_VISIBLE_DEVICES=0).  Each job is short (~2-3 min).
The script never kills sibling processes; it just waits until GPU free.

Outputs are stored under
  outputs/sweep_censored_stage_survival_anti_plip_vec_LUAD/{tag}/fold_{f}_seed_{s}/
with tag in {baseline, anti_on}.  At the end writes
  outputs/sweep_censored_stage_survival_anti_plip_vec_LUAD/aggregate.json
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_censored_stage_survival.py"
SPLIT_ROOT = ROOT / "data" / "splits" / "censored_stage_protocol" / "phase1_outer5"
SWEEP_DIR = ROOT / "outputs" / "sweep_censored_stage_survival_anti_plip_vec_LUAD"
ANTI_DIR = ROOT / "outputs" / "anti_feature_experiment"

COMMON = [
    "--stage", "1",
    "--target_col", "os_survival_days",
    "--wsi_feature_source", "plip_luad_256",
    "--rna_mode", "vec",
    "--epochs", "150", "--batch_size", "16", "--max_tiles", "4096",
    "--hidden_dim", "256", "--lr", "1e-3", "--weight_decay", "1e-4",
    "--model_variant", "baseline", "--pool_method", "attention",
    "--cross_modal_fusion", "concat",
    "--loss_window_lower", "0.59", "--loss_window_upper", "0.72",
    "--loss_window_k", "50.0", "--loss_window_A", "0.99",
    "--num_workers", "2", "--pin_memory",
    "--device", "cuda",
]
ANTI_EXTRA = [
    "--use-anti-injection",
    "--anti-feature-dir", str(ANTI_DIR),
    "--anti-feature-cohorts", "LUAD",
]
FOLDS = list(range(5))
SEEDS = list(range(3))
TAGS = [("baseline", []), ("anti_on", ANTI_EXTRA)]


@dataclass
class Job:
    tag: str
    fold: int
    seed: int
    extra: list[str]

    @property
    def name(self) -> str:
        return f"{self.tag}__fold{self.fold}_seed{self.seed}"

    @property
    def out_dir(self) -> Path:
        return SWEEP_DIR / self.tag / f"fold_{self.fold}_seed_{self.seed}"


def make_jobs() -> list[Job]:
    jobs: list[Job] = []
    for tag, extra in TAGS:
        for f in FOLDS:
            for s in SEEDS:
                jobs.append(Job(tag=tag, fold=f, seed=s, extra=list(extra)))
    # Order: all baselines first?  Interleave so early runs are comparable.
    jobs.sort(key=lambda j: (j.fold, j.seed, j.tag))
    return jobs


def run_job(job: Job, *, idx: int, total: int, env_add: dict[str, str] | None = None) -> None:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = str(SPLIT_ROOT / f"fold_{job.fold}")
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        *COMMON,
        "--split_dir", split_dir,
        "--seed", str(job.seed),
        "--out_dir", str(job.out_dir),
        *job.extra,
    ]
    log_path = job.out_dir / "run.log"
    env = dict(**subprocess.os.environ)
    if env_add:
        env.update(env_add)
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    env.setdefault("NUMEXPR_MAX_THREADS", "8")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time()
    print(f"[sweep {idx+1}/{total}] {datetime.now().isoformat(timespec='seconds')} "
          f"START {job.name}  -> {job.out_dir}", flush=True)
    with log_path.open("wb") as f:
        p = subprocess.run(cmd, env=env, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    print(f"[sweep {idx+1}/{total}] {datetime.now().isoformat(timespec='seconds')} "
          f"DONE  {job.name}  {status}  elapsed={dt:.0f}s", flush=True)


def load_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def aggregate(jobs: list[Job]) -> dict[str, object]:
    rows = []
    headers = ["tag", "fold", "seed", "exit", "best_epoch", "best_val_c_index_ema",
               "best_val_c_index", "best_test_c_index", "best_train_c_index",
               "final_test_c_index", "n_train_cases", "n_val_cases", "n_test_cases"]
    by_pair: dict[tuple[int, int], dict[str, float]] = {}
    for job in jobs:
        s = load_summary(job.out_dir / "summary.json")
        exit_code = "OK" if (job.out_dir / "summary.json").is_file() and s else "FAIL"
        row = {
            "tag": job.tag,
            "fold": job.fold,
            "seed": job.seed,
            "exit": exit_code,
            "best_epoch": s.get("best_epoch"),
            "best_val_c_index_ema": s.get("best_select_score"),
            "best_val_c_index": s.get("best_val_c_index"),
            "best_test_c_index": s.get("best_test_c_index"),
            "best_train_c_index": s.get("best_train_c_index"),
            "final_test_c_index": s.get("final_test_c_index"),
            "n_train_cases": s.get("n_train_cases"),
            "n_val_cases": s.get("n_val_cases"),
            "n_test_cases": s.get("n_test_cases"),
        }
        rows.append(row)
        key = (job.fold, job.seed)
        by_pair.setdefault(key, {})[job.tag] = row["best_test_c_index"]
    # Per-pair deltas
    deltas_test = []
    for key, m in by_pair.items():
        if isinstance(m.get("anti_on"), float) and isinstance(m.get("baseline"), float):
            deltas_test.append(m["anti_on"] - m["baseline"])
    import numpy as np
    dt = np.asarray(deltas_test, dtype=np.float64)
    agg = {
        "sweep_root": str(SWEEP_DIR),
        "n_jobs_total": len(jobs),
        "n_jobs_ok": sum(1 for r in rows if r["exit"] == "OK"),
        "paired_runs": len(deltas_test),
        "delta_best_test_c_index_mean": float(np.mean(dt)) if dt.size else None,
        "delta_best_test_c_index_std": float(np.std(dt)) if dt.size else None,
        "delta_best_test_c_index_pct_positive": float(np.mean(dt > 0) * 100.0) if dt.size else None,
        "rows": rows,
        "headers": headers,
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    csv_path = SWEEP_DIR / "aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    return agg


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs()
    print(f"[sweep] launching {len(jobs)} jobs serially...", flush=True)
    # Skip jobs that are already OK (re-runnable)
    pending: list[tuple[int, Job]] = []
    for i, j in enumerate(jobs):
        if (j.out_dir / "summary.json").is_file():
            print(f"[sweep skip {i+1}/{len(jobs)}] already done: {j.name}", flush=True)
        else:
            pending.append((i, j))
    print(f"[sweep] pending jobs: {len(pending)}", flush=True)
    try:
        for i, j in pending:
            try:
                run_job(j, idx=i, total=len(jobs))
            except Exception as exc:  # pragma: no cover
                print(f"[sweep exception] {j.name}: {exc}", flush=True)
    finally:
        agg = aggregate(jobs)
        done = agg["n_jobs_ok"]; tot = agg["n_jobs_total"]
        print(f"\n[sweep DONE] {done}/{tot} jobs succeeded", flush=True)
        if agg["paired_runs"]:
            print(f"  Δ best_test_c_index (anti_on - baseline) n={agg['paired_runs']}: "
                  f"{agg['delta_best_test_c_index_mean']*100:+.2f} ± "
                  f"{agg['delta_best_test_c_index_std']*100:.2f} pp")
            print(f"  % runs anti_on > baseline : {agg['delta_best_test_c_index_pct_positive']:.1f}%")
        print(f"  aggregate.json : {SWEEP_DIR / 'aggregate.json'}")
        print(f"  aggregate.csv  : {SWEEP_DIR / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
