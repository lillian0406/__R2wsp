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
PHASE2_OUT_ROOT = PROJECT_ROOT / "outputs" / "censored_stage_survival_phase2" / "k3_grid_window" / "seed0"
SPLIT_P2_ROOT = PROJECT_ROOT / "data" / "splits" / "censored_stage_protocol" / "phase2_independent5"

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
    "--stage", "2", "--seed", "0", "--epochs", "50",
]

FOLDS_K3 = [0, 1, 2]
GRID = [
    (0.59, 0.65), (0.59, 0.68), (0.59, 0.72),
    (0.55, 0.65), (0.55, 0.68), (0.55, 0.72),
    (0.50, 0.65), (0.50, 0.68), (0.50, 0.72),
]


@dataclass
class Plan:
    tag: str
    cmd: list[str]
    out_dir: Path
    L: float
    U: float
    fold: int


def build_plan(fold: int, L: float, U: float, A: float = 0.99, K: float = 50.0,
               eps: float = 1e-3, metric: str = "val_c_index_ema") -> Plan:
    win_tag = f"win{L:.2f}-{U:.2f}_A{A:.2f}_K{K:.0f}"
    split_dir = SPLIT_P2_ROOT / f"fold_{fold}" / "eval_with_censored"
    pretrain = PHASE1_OUT / f"fold_{fold}_seed0" / "best.pt"
    out_dir = PHASE2_OUT_ROOT / win_tag / f"fold_{fold}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *BASE_ARGS,
        "--split_dir", str(split_dir),
        "--pretrain_checkpoint", str(pretrain),
        "--out_dir", str(out_dir),
        "--loss_window_lower", f"{L}",
        "--loss_window_upper", f"{U}",
        "--loss_window_k", f"{K}",
        "--loss_window_A", f"{A}",
        "--loss_window_eps", f"{eps}",
        "--loss_window_metric", metric,
    ]
    return Plan(tag=f"{win_tag}_f{fold}", cmd=cmd, out_dir=out_dir, L=L, U=U, fold=fold)


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
    rows: list[tuple[str, float, float, list[float], list[int | str]]] = []
    for (L, U) in GRID:
        win_tag = f"win{L:.2f}-{U:.2f}_A0.99_K50"
        vals: list[float] = []
        eps: list[int | str] = []
        n_ok = 0
        for f in FOLDS_K3:
            sj = PHASE2_OUT_ROOT / win_tag / f"fold_{f}" / "summary.json"
            if not sj.exists():
                continue
            j = json.load(open(sj))
            vals.append(j.get("final_test_c_index", float("nan")))
            eps.append(j.get("best_epoch", "?"))
            n_ok += 1
        rows.append((win_tag, L, U, vals, eps))
    rows.sort(key=lambda r: (st.mean(r[3]) if r[3] else -1), reverse=True)

    lines: list[str] = []
    lines.append("=== 3 折 seed0 窗口超参小网格（按均值排序，9 组）===")
    lines.append(f"{'排名':>4s}  {'窗口':30s}  {'mean±std':16s}  {'n':>2s}  ep_mode   fold_0/1/2")
    for i, (tag, L, U, vals, eps) in enumerate(rows, 1):
        if not vals:
            lines.append(f"{i:>4d}  {tag:30s}  {'未完成':16s}  {len(vals):>2d}")
            continue
        m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        mode_ep = max(set(eps), key=eps.count) if eps else "?"
        cells = "  ".join(f"{v:.3f}(e{e})" for v, e in zip(vals, eps))
        lines.append(f"{i:>4d}  {tag:30s}  {m:.4f}±{sd:.4f}  {len(vals):>2d}  {str(mode_ep):>6s}   {cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-report", action="store_true", help="只做现有结果汇总，不跑")
    ap.add_argument("--no-skip", action="store_true", help="覆盖重跑已有结果")
    args = ap.parse_args()
    plans = [build_plan(f, L, U) for (L, U) in GRID for f in FOLDS_K3]
    print(f"[INFO] planned {len(plans)} runs  (3 folds * 9 windows, seed0 fixed)")
    if not args.only_report:
        for p in plans:
            rc = run_one(p, skip_existing=not args.no_skip)
            if rc != 0:
                raise SystemExit(f"[FAIL] {p.tag} rc={rc}")
    rep = summary_report(plans)
    print("\n" + rep)
    out_txt = PHASE2_OUT_ROOT / "summary_grid.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(rep + "\n")
    print(f"\n[save] {out_txt}")


if __name__ == "__main__":
    main()
