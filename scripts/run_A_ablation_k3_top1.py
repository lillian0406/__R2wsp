#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/root/autodl-tmp/venvs/tcga1126/bin/python").resolve()
SCRIPT = PROJECT_ROOT / "scripts" / "train_censored_stage_survival.py"
CONFIG = PROJECT_ROOT / "configs" / "data_paths.yaml"
PHASE1_OUT = PROJECT_ROOT / "outputs" / "censored_stage_survival_phase1"
OUT_ROOT_A = PROJECT_ROOT / "outputs" / "censored_stage_survival_phase2" / "k3_grid_A_ablation" / "top1_win0.59-0.68_seed0"
SPLIT_P2_ROOT = PROJECT_ROOT / "data" / "splits" / "censored_stage_protocol" / "phase2_independent5"

# 主实验：top-1 窗口（L=0.59, U=0.68）× A ∈ {0.99, 0.995, 1.00, 1.005}，3 折 seed0
FIXED_L = 0.59
FIXED_U = 0.68
FIXED_K = 50.0
FIXED_EPS = 1e-3
FIXED_METRIC = "val_c_index_ema"
A_VALUES = [0.99, 0.995, 1.00, 1.005]
FOLDS_K3 = [0, 1, 2]
SEED = 0
EPOCHS = 50

BASE_ARGS = [
    "--config", str(CONFIG), "--data_root", "data",
    "--wsi_feature_source", "plip_luad_256", "--target_col", "dss_survival_days",
    "--max_tiles", "256", "--batch_size", "8",
    "--lr", "1e-4", "--weight_decay", "1e-4",
    "--val_frac", "0.2", "--val_split_mode", "survival_stratified",
    "--selection_min_epochs", "8", "--early_stop_patience", "12",
    "--hidden_dim", "256", "--dropout", "0.15",
    "--model_variant", "baseline", "--fusion_mode", "concat",
    "--num_workers", "2", "--pin_memory",
    "--stage", "2", "--seed", str(SEED), "--epochs", str(EPOCHS),
]


@dataclass
class Plan:
    tag: str
    cmd: list[str]
    out_dir: Path
    A: float
    fold: int


def build_plan(fold: int, A: float) -> Plan:
    win_tag = f"win{FIXED_L:.2f}-{FIXED_U:.2f}_A{A:.3f}_K{FIXED_K:.0f}"
    split_dir = SPLIT_P2_ROOT / f"fold_{fold}" / "eval_with_censored"
    pretrain = PHASE1_OUT / f"fold_{fold}_seed{SEED}" / "best.pt"
    out_dir = OUT_ROOT_A / win_tag / f"fold_{fold}"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *BASE_ARGS,
        "--split_dir", str(split_dir),
        "--pretrain_checkpoint", str(pretrain),
        "--out_dir", str(out_dir),
        "--loss_window_lower", f"{FIXED_L}",
        "--loss_window_upper", f"{FIXED_U}",
        "--loss_window_k", f"{FIXED_K}",
        "--loss_window_A", f"{A}",
        "--loss_window_eps", f"{FIXED_EPS}",
        "--loss_window_metric", FIXED_METRIC,
    ]
    return Plan(tag=f"{win_tag}_f{fold}", cmd=cmd, out_dir=out_dir, A=A, fold=fold)


def _exists(out_dir: Path) -> bool:
    return (out_dir / "final.pt").exists() and (out_dir / "summary.json").exists()


def run_one(p: Plan, skip_existing: bool = True) -> int:
    if skip_existing and _exists(p.out_dir):
        print(f"[SKIP] {p.tag}", flush=True)
        return 0
    p.out_dir.mkdir(parents=True, exist_ok=True)
    (p.out_dir / "histories").mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN ] {p.tag}  out={p.out_dir}", flush=True)
    with (p.out_dir / "run.log").open("ab") as log_f:
        proc = subprocess.Popen(
            p.cmd, cwd=str(PROJECT_ROOT),
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
    print(f"[DONE] {p.tag}  rc={rc}", flush=True)
    return rc


def summary_report(plans: list[Plan]) -> str:
    rows: list[tuple[str, float, list[float], list[str | int]]] = []
    for A in A_VALUES:
        win_tag = f"win{FIXED_L:.2f}-{FIXED_U:.2f}_A{A:.3f}_K{FIXED_K:.0f}"
        vals: list[float] = []
        eps: list[str | int] = []
        for f in FOLDS_K3:
            sj = OUT_ROOT_A / win_tag / f"fold_{f}" / "summary.json"
            if not sj.exists():
                continue
            j = json.load(open(sj))
            vals.append(j.get("final_test_c_index", float("nan")))
            eps.append(j.get("best_epoch", "?"))
        rows.append((win_tag, A, vals, eps))
    rows.sort(key=lambda r: (st.mean(r[2]) if r[2] else -1), reverse=True)

    lines: list[str] = []
    lines.append(f"=== A 参数天花板验证  窗口 L={FIXED_L} U={FIXED_U}  k3 seed0 （A ∈ {A_VALUES}）===")
    lines.append(f"{'排名':>4s}  {'窗口/Win Tag':36s}  {'A':>5s}  {'mean±std':16s}  n  ep_mode    fold_0/1/2")
    for i, (tag, A, vals, eps) in enumerate(rows, 1):
        if not vals:
            lines.append(f"{i:>4d}  {tag:36s}  {A:1.3f}  {'未完成':16s}  {len(vals)}")
            continue
        m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        mode_ep = max(set(eps), key=eps.count) if eps else "?"
        cells = "  ".join(f"{v:.3f}(e{e})" for v, e in zip(vals, eps))
        lines.append(f"{i:>4d}  {tag:36s}  {A:1.3f}  {m:.4f}±{sd:.4f}  {len(vals)}  {str(mode_ep):>6s}    {cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()
    plans = [build_plan(f, A) for A in A_VALUES for f in FOLDS_K3]
    print(f"[INFO] planned {len(plans)} runs  (4 A × 3 folds, seed0, win L={FIXED_L} U={FIXED_U})")
    if not args.only_report:
        for p in plans:
            rc = run_one(p, skip_existing=not args.no_skip)
            if rc != 0:
                raise SystemExit(f"[FAIL] {p.tag} rc={rc}")
    rep = summary_report(plans)
    print("\n" + rep)
    out_txt = OUT_ROOT_A / "summary_A_ablation.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(rep + "\n")
    print(f"\n[save] {out_txt}")


if __name__ == "__main__":
    main()
