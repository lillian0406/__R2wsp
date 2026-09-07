#!/usr/bin/env python3
"""
Task A: Phase-2 sweep (censored included, windowed Cox)
========================================================
For each (fold, seed) in phase1_outer5 folds 0..4 x seeds 0..2:
  (a) Run stage-2 baseline:  pretrain = phase1 baseline best.pt
  (b) Run stage-2 anti_on :  pretrain = phase1 anti_on  best.pt
Total 30 jobs, serial on GPU-0.  Uses scripts/train_censored_stage_survival.py --stage 2.

Window loss matches full scheduler (no-touch L49 constants):
  loss_window_lower=0.59 / upper=0.72 / k=50 / A=0.99 / eps=1e-3 (CLI defaults adjusted here).

Phase-2 runs train N epochs=50 (shorter, per window_loss protocol).
Outputs:
  outputs/sweep_censored_stage2_survival_anti_plip_vec_LUAD/{baseline,anti_on}/fold_{f}_seed_{s}/
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_censored_stage_survival.py"
SPLIT_ROOT = ROOT / "data" / "splits" / "censored_stage_protocol" / "phase1_outer5"
ANTI_DIR = ROOT / "outputs" / "anti_feature_experiment"

PHASE1 = ROOT / "outputs" / "sweep_censored_stage_survival_anti_plip_vec_LUAD"
SWEEP2 = ROOT / "outputs" / "sweep_censored_stage2_survival_anti_plip_vec_LUAD"

COMMON = [
    "--stage", "2",
    "--target_col", "os_survival_days",
    "--wsi_feature_source", "plip_luad_256",
    "--rna_mode", "vec",
    "--epochs", "50", "--batch_size", "16", "--max_tiles", "4096",
    "--hidden_dim", "256", "--lr", "5e-4", "--weight_decay", "1e-4",
    "--model_variant", "baseline", "--pool_method", "attention",
    "--cross_modal_fusion", "concat",
    "--loss_window_lower", "0.59", "--loss_window_upper", "0.72",
    "--loss_window_k", "50.0", "--loss_window_A", "0.99",
    "--loss_window_eps", "1e-3", "--loss_window_metric", "val_c_index_ema",
    "--num_workers", "2", "--pin_memory",
    "--device", "cuda",
]

TAGS = [
    ("baseline", []),
    ("anti_on", [
        "--use-anti-injection",
        "--anti-feature-dir", str(ANTI_DIR),
        "--anti-feature-cohorts", "LUAD",
    ]),
]

FOLDS = list(range(5))
SEEDS = list(range(3))


@dataclass
class Job:
    tag: str
    fold: int
    seed: int
    extra: list[str]

    @property
    def name(self) -> str:
        return f"s2_{self.tag}__fold{self.fold}_seed{self.seed}"

    @property
    def phase1_best_pt(self) -> Path:
        # Stage-2 pretrain checkpoint = phase1 best.pt of the SAME (tag, fold, seed)
        return PHASE1 / self.tag / f"fold_{self.fold}_seed_{self.seed}" / "best.pt"

    @property
    def out_dir(self) -> Path:
        return SWEEP2 / self.tag / f"fold_{self.fold}_seed_{self.seed}"


def make_jobs() -> list[Job]:
    jobs: list[Job] = []
    for tag, extra in TAGS:
        for f in FOLDS:
            for s in SEEDS:
                jobs.append(Job(tag=tag, fold=f, seed=s, extra=list(extra)))
    jobs.sort(key=lambda j: (j.fold, j.seed, j.tag))
    return jobs


def run_job(job: Job, *, idx: int, total: int) -> None:
    if not job.phase1_best_pt.is_file():
        raise FileNotFoundError(f"Phase1 best.pt missing: {job.phase1_best_pt}")
    job.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = str(SPLIT_ROOT / f"fold_{job.fold}")
    cmd = [
        sys.executable, str(TRAIN),
        *COMMON,
        "--split_dir", split_dir,
        "--seed", str(job.seed),
        "--pretrain_checkpoint", str(job.phase1_best_pt),
        "--out_dir", str(job.out_dir),
        *job.extra,
    ]
    log_path = job.out_dir / "run.log"
    env = dict(**subprocess.os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    env.setdefault("NUMEXPR_MAX_THREADS", "8")
    t0 = time.time()
    print(f"[s2-sweep {idx+1}/{total}] {datetime.now().isoformat(timespec='seconds')} START {job.name}", flush=True)
    with log_path.open("wb") as f:
        p = subprocess.run(cmd, env=env, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
    print(f"[s2-sweep {idx+1}/{total}] {datetime.now().isoformat(timespec='seconds')} DONE  {job.name}  {status}  elapsed={dt:.0f}s", flush=True)


def aggregate(jobs: list[Job]) -> dict:
    import numpy as np
    rows = []
    pairs: dict[tuple[int, int], dict[str, float]] = {}
    for job in jobs:
        s = {}
        sp = job.out_dir / "summary.json"
        if sp.is_file():
            s = json.loads(sp.read_text())
        ok = bool(s)
        rows.append({
            "tag": job.tag, "fold": job.fold, "seed": job.seed, "ok": ok,
            "last_epoch": s.get("last_epoch"),
            "last_val_c_ema": s.get("last_val_c_index_ema"),
            "last_val_c": s.get("last_val_c_index"),
            "final_test_c": s.get("final_test_c_index"),
            "window_activated": s.get("window_loss_activated"),
            "window_w_now_last": s.get("window_w_now_last"),
        })
        if ok:
            pairs.setdefault((job.fold, job.seed), {})[job.tag] = s.get("final_test_c_index")
    deltas = []
    for m in pairs.values():
        if isinstance(m.get("anti_on"), float) and isinstance(m.get("baseline"), float):
            deltas.append(m["anti_on"] - m["baseline"])
    arr = np.asarray(deltas) if deltas else np.zeros(0)
    agg = {
        "sweep_root": str(SWEEP2),
        "n_jobs_total": len(jobs),
        "n_jobs_ok": sum(1 for r in rows if r["ok"]),
        "paired_runs": len(deltas),
        "delta_final_test_c_mean_pp": float(arr.mean() * 100) if arr.size else None,
        "delta_final_test_c_std_pp": float(arr.std(ddof=1) * 100) if arr.size else None,
        "delta_final_test_c_pct_positive": float((arr > 0).mean() * 100) if arr.size else None,
        "rows": rows,
    }
    SWEEP2.mkdir(parents=True, exist_ok=True)
    (SWEEP2 / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    with (SWEEP2 / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return agg


def main() -> None:
    SWEEP2.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs()
    pending: list[tuple[int, Job]] = []
    for i, j in enumerate(jobs):
        if (j.out_dir / "summary.json").is_file():
            print(f"[s2-sweep skip {i+1}/{len(jobs)}] already done: {j.name}", flush=True)
        else:
            pending.append((i, j))
    print(f"[s2-sweep] launching {len(pending)}/{len(jobs)} pending jobs serially", flush=True)
    try:
        for i, j in pending:
            try:
                run_job(j, idx=i, total=len(jobs))
            except Exception as exc:
                print(f"[s2-sweep exception] {j.name}: {exc}", flush=True)
    finally:
        agg = aggregate(jobs)
        print(f"\n[s2-sweep DONE] {agg['n_jobs_ok']}/{agg['n_jobs_total']} jobs succeeded", flush=True)
        if agg["paired_runs"]:
            print(f"  Δfinal_test_c_index (anti_on - baseline) paired n={agg['paired_runs']}: "
                  f"{agg['delta_final_test_c_mean_pp']:+.2f} ± "
                  f"{agg['delta_final_test_c_std_pp']:.2f} pp  "
                  f"positive={agg['delta_final_test_c_pct_positive']:.1f}%")
        print(f"  aggregate.json : {SWEEP2 / 'aggregate.json'}")
        print(f"  aggregate.csv  : {SWEEP2 / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
