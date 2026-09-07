#!/usr/bin/env python3
"""删失分阶段实验调度器：Phase-1 5折×3seed → Phase-2 5折×3seed（串行、独立日志、失败即停）。
执行前请确认 UNI 特征抽取进程不在高 GPU 占用，否则会互相拖慢。
两套 eval 口径默认跑 eval_with_censored，如需 eval_uncensored_only 对照可改 EVAL_KIND。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/root/autodl-tmp/venvs/tcga1126/bin/python").resolve()
SCRIPT = PROJECT_ROOT / "scripts" / "train_censored_stage_survival.py"
CONFIG = PROJECT_ROOT / "configs" / "data_paths.yaml"
DATA_ROOT = "data"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits" / "censored_stage_protocol"
PHASE1_ROOT = SPLITS_ROOT / "phase1_outer5"
PHASE2_ROOT = SPLITS_ROOT / "phase2_independent5"
OUT_ROOT = PROJECT_ROOT / "outputs"
PHASE1_OUT = OUT_ROOT / "censored_stage_survival_phase1"
PHASE2_OUT = OUT_ROOT / "censored_stage_survival_phase2"

# no-geo 基线，严格对齐协议：
BASE_ARGS = [
    "--config", str(CONFIG),
    "--data_root", DATA_ROOT,
    "--wsi_feature_source", "plip_luad_256",
    "--target_col", "dss_survival_days",
    "--max_tiles", "256",
    "--batch_size", "8",
    "--lr", "1e-4",
    "--weight_decay", "1e-4",
    "--val_frac", "0.2",
    "--val_split_mode", "survival_stratified",
    "--selection_min_epochs", "8",
    "--early_stop_patience", "12",
    "--hidden_dim", "256",
    "--dropout", "0.15",
    "--model_variant", "baseline",
    "--fusion_mode", "concat",
    "--num_workers", "2",
    "--pin_memory",
]

# 窗口参数（stage=2 生效，A=0.99 默认；A=1 对照可改 WINDOW_A）
WINDOW_LOWER = 0.59
WINDOW_UPPER = 0.68
WINDOW_K = 50.0
WINDOW_A = 0.99
WINDOW_EPS = 1e-3
WINDOW_METRIC = "val_c_index_ema"
_WIN_TAG = f"win{WINDOW_LOWER:.2f}-{WINDOW_UPPER:.2f}_A{WINDOW_A:.2f}_K{WINDOW_K:.0f}"

N_FOLDS = 5
_FOLD_TAG = f"k{N_FOLDS}"
SEEDS = [0, 1, 2]
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50
EVAL_KIND = "eval_with_censored"  # "eval_with_censored" 或 "eval_uncensored_only"


@dataclass
class RunPlan:
    tag: str
    cmd: list[str]
    out_dir: Path
    log_file: Path


def build_phase1_plan(fold: int, seed: int) -> RunPlan:
    split_dir = PHASE1_ROOT / f"fold_{fold}"
    out_dir = PHASE1_OUT / f"fold_{fold}_seed{seed}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *BASE_ARGS,
        "--split_dir", str(split_dir),
        "--stage", "1",
        "--epochs", str(EPOCHS_PHASE1),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]
    return RunPlan(tag=f"P1_f{fold}_s{seed}", cmd=cmd, out_dir=out_dir, log_file=log_file)


def build_phase2_plan(fold: int, seed: int, eval_kind: str = EVAL_KIND) -> RunPlan:
    split_dir = PHASE2_ROOT / f"fold_{fold}" / eval_kind
    pretrain = PHASE1_OUT / f"fold_{fold}_seed{seed}" / "best.pt"
    out_dir = PHASE2_OUT / _FOLD_TAG / _WIN_TAG / eval_kind / f"fold_{fold}_seed{seed}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *BASE_ARGS,
        "--split_dir", str(split_dir),
        "--stage", "2",
        "--pretrain_checkpoint", str(pretrain),
        "--epochs", str(EPOCHS_PHASE2),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
        "--loss_window_lower", str(WINDOW_LOWER),
        "--loss_window_upper", str(WINDOW_UPPER),
        "--loss_window_k", str(WINDOW_K),
        "--loss_window_A", str(WINDOW_A),
        "--loss_window_eps", str(WINDOW_EPS),
        "--loss_window_metric", WINDOW_METRIC,
    ]
    return RunPlan(tag=f"P2_{eval_kind[:4]}_{_FOLD_TAG}_{_WIN_TAG}_f{fold}_s{seed}", cmd=cmd, out_dir=out_dir, log_file=log_file)


def run_one(plan: RunPlan, skip_existing: bool = False) -> int:
    def _marker():
        if (plan.out_dir / "final.pt").exists() and (plan.out_dir / "summary.json").exists():
            return plan.out_dir / "final.pt"
        if (plan.out_dir / "best.pt").exists() and (plan.out_dir / "summary.json").exists():
            return plan.out_dir / "best.pt"
        return None
    m = _marker()
    if skip_existing and m is not None:
        print(f"\n[SKIP] {plan.tag}  marker={m.name} 已存在，跳过", flush=True)
        return 0
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    (plan.out_dir / "histories").mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN ] {plan.tag}  out={plan.out_dir}", flush=True)
    with plan.log_file.open("ab") as log_f:
        proc = subprocess.Popen(
            plan.cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={"CUDA_VISIBLE_DEVICES": "0", **dict(__import__("os").environ)},
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            sys.stdout.buffer.write(raw)
            sys.stdout.flush()
            log_f.write(raw)
            log_f.flush()
        rc = proc.wait()
    print(f"[DONE] {plan.tag}  rc={rc}", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["phase1", "phase2", "all"], default="all")
    ap.add_argument("--start-from", default=None,
                    help="从某个 tag 开始（包含），例如 P1_f2_s1 。用于中断后恢复。")
    ap.add_argument("--eval-kind", choices=["eval_with_censored", "eval_uncensored_only"], default=EVAL_KIND)
    ap.add_argument("--skip-existing", action="store_true",
                    help="如果 best.pt/final.pt + summary.json 已存在则跳过。用于 5 折补 fold_3/4 的 Phase-1。")
    args = ap.parse_args()

    eval_kind = str(args.eval_kind)
    skip_existing = bool(args.skip_existing)

    plans: list[RunPlan] = []
    if args.only in ("phase1", "all"):
        for f in range(N_FOLDS):
            for s in SEEDS:
                plans.append(build_phase1_plan(f, s))
    if args.only in ("phase2", "all"):
        for f in range(N_FOLDS):
            for s in SEEDS:
                plans.append(build_phase2_plan(f, s, eval_kind))

    if args.start_from:
        idx = next((i for i, p in enumerate(plans) if p.tag == args.start_from), None)
        if idx is None:
            raise SystemExit(f"start-from {args.start_from} 不在计划中：{[p.tag for p in plans]}")
        plans = plans[idx:]

    print(f"[INFO] N_FOLDS={N_FOLDS} _FOLD_TAG={_FOLD_TAG} _WIN_TAG={_WIN_TAG}")
    print(f"[INFO] total {len(plans)} runs planned（first={plans[0].tag if plans else '-'}, last={plans[-1].tag if plans else '-'}）")
    for p in plans:
        rc = run_one(p, skip_existing=skip_existing)
        if rc != 0:
            raise SystemExit(f"[FAIL] {p.tag} exit={rc}. 恢复可加 --start-from {p.tag}")
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
