#!/usr/bin/env python3
"""BRCA UNI1024 + Hallmark50 两阶段 full 调度器：5 折 × 3 seeds × 2 eval_kind × 1 exp = 45 runs。
严格对齐主 scheduler（run_full_upgrade_k5x3seeds.py）结构，关键差异：
  - cohort_hint=BRCA
  - split_dir=data/splits/censored_stage_protocol_BRCA/
  - out=outputs/censored_stage_survival_full_upgrade_BRCA/...
  - WSI source：主 = uni1024_tilefix256_spatial，aliases=uni1024（解决混目录问题）
  - RNA mode：omics（symbol TSV 已软链到 tpm_tsv/brca_symbol_tpm.tsv）
  - Window params：对齐 LUAD 主链路 0.59-0.65_A0.99_K50（不是摘要里 0.68，论文正文写 0.65 区间）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import statistics as st
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/root/miniconda3/bin/python3.12").resolve()
SCRIPT = PROJECT_ROOT / "scripts" / "train_censored_stage_survival.py"
CONFIG = PROJECT_ROOT / "configs" / "data_paths.yaml"
DATA_ROOT = "data"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits" / "censored_stage_protocol_BRCA"
PHASE1_ROOT = SPLITS_ROOT / "phase1_outer5"
PHASE2_ROOT = SPLITS_ROOT / "phase2_independent5"
OUT_ROOT = PROJECT_ROOT / "outputs" / "censored_stage_survival_full_upgrade_BRCA"

WINDOW_LOWER = 0.59
WINDOW_UPPER = 0.65
WINDOW_K = 50.0
WINDOW_A = 0.99
WINDOW_EPS = 1e-3
WINDOW_METRIC = "val_c_index_ema"
_WIN_TAG = f"win{WINDOW_LOWER:.2f}-{WINDOW_UPPER:.2f}_A{WINDOW_A:.2f}_K{WINDOW_K:.0f}"

N_FOLDS = 5
SEEDS = [0, 1, 2]
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50
EVAL_KINDS = ["eval_with_censored", "eval_uncensored_only"]
BASELINE_PLI_VEC = {"eval_with_censored": None, "eval_uncensored_only": None}

GTF_PATH = "/root/autodl-tmp/gencode.v22.annotation.gtf.gz"


@dataclass(frozen=True)
class ExpSpec:
    feat_tag: str
    wsi_feature_source: str
    wsi_feature_source_aliases: str  # 逗号分隔
    rna_mode: str = "omics"
    batch_size: int = 4


EXPERIMENTS: list[ExpSpec] = [
    ExpSpec(
        feat_tag="wsi=UNI1024_tilefix256_rna=hallmark50_omics",
        wsi_feature_source="uni1024_tilefix256_spatial",
        wsi_feature_source_aliases="uni1024,uni1024_tilefix256_spatial",
    ),
]


@dataclass
class RunPlan:
    tag: str
    cmd: list[str]
    out_dir: Path
    log_file: Path
    cuda: str  # CUDA_VISIBLE_DEVICES


def build_base_args(exp: ExpSpec) -> list[str]:
    return [
        "--config", str(CONFIG),
        "--data_root", DATA_ROOT,
        "--wsi_feature_source", exp.wsi_feature_source,
        "--wsi_feature_source_aliases", exp.wsi_feature_source_aliases,
        "--target_col", "dss_survival_days",
        "--max_tiles", "256",
        "--batch_size", str(exp.batch_size),
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
        "--rna_mode", exp.rna_mode,
        "--gene_annotation_gtf", GTF_PATH,
        "--cohort_hint", "BRCA",
    ]


def build_phase1_plan(exp: ExpSpec, fold: int, seed: int, cuda: str) -> RunPlan:
    split_dir = PHASE1_ROOT / f"fold_{fold}"
    out_dir = OUT_ROOT / exp.feat_tag / _WIN_TAG / "phase1" / f"fold_{fold}_seed{seed}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *build_base_args(exp),
        "--split_dir", str(split_dir),
        "--stage", "1",
        "--epochs", str(EPOCHS_PHASE1),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]
    return RunPlan(tag=f"[P1]{exp.feat_tag[:20]}.._f{fold}_s{seed}", cmd=cmd, out_dir=out_dir, log_file=log_file, cuda=cuda)


def build_phase2_plan(exp: ExpSpec, fold: int, seed: int, eval_kind: str, cuda: str) -> RunPlan:
    split_dir = PHASE2_ROOT / f"fold_{fold}" / eval_kind
    pretrain = OUT_ROOT / exp.feat_tag / _WIN_TAG / "phase1" / f"fold_{fold}_seed{seed}" / "best.pt"
    out_dir = OUT_ROOT / exp.feat_tag / _WIN_TAG / eval_kind / "phase2" / f"fold_{fold}_seed{seed}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *build_base_args(exp),
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
    return RunPlan(tag=f"[P2]{exp.feat_tag[:20]}.._f{fold}_s{seed}_{eval_kind[:4]}", cmd=cmd, out_dir=out_dir, log_file=log_file, cuda=cuda)


def _marker_exists(out_dir: Path) -> bool:
    ok_ckpt = (out_dir / "final.pt").exists() or (out_dir / "best.pt").exists()
    ok_summ = (out_dir / "summary.json").exists()
    return bool(ok_ckpt and ok_summ)


def run_one(plan: RunPlan, skip_existing: bool = True) -> int:
    if skip_existing and _marker_exists(plan.out_dir):
        print(f"\n[SKIP] {plan.tag}  summary.json + best.pt 已存在 → 跳过", flush=True)
        return 0
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    (plan.out_dir / "histories").mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN ] {plan.tag}  (CUDA={plan.cuda})", flush=True)
    print(f"       out = {plan.out_dir}", flush=True)
    with plan.log_file.open("ab") as log_f:
        env = dict(__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = plan.cuda
        proc = subprocess.Popen(
            plan.cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env,
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


def _final_c(summary_json: Path) -> float | None:
    if not summary_json.exists():
        return None
    try:
        j = json.load(open(summary_json))
    except Exception:
        return None
    for k in ("final_test_c_index", "best_test_c_index", "test_c_index_ema", "test_c_index"):
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def summary_report() -> str:
    lines: list[str] = []
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    lines.append("=" * 120)
    lines.append(f"BRCA FULL UPGRADE 汇总报告  ({t0})   窗口={_WIN_TAG}  k={N_FOLDS}  seeds={SEEDS}")
    lines.append(f"Experiment: {EXPERIMENTS[0].feat_tag}  (WSI aliases={EXPERIMENTS[0].wsi_feature_source_aliases})")
    lines.append("")
    for eval_kind in EVAL_KINDS:
        baseline_ref = BASELINE_PLI_VEC.get(eval_kind)
        bl = f"Baseline ref = {baseline_ref:.4f}" if baseline_ref else f"eval_kind={eval_kind} (no baseline ref yet — LUAD=0.712±0.049)"
        lines.append(f"--- {eval_kind} ---   {bl}")
        rows: list[tuple[str, list[float], list[int]]] = []
        for exp in EXPERIMENTS:
            vals: list[float] = []
            eps: list[int] = []
            for f in range(N_FOLDS):
                for s in SEEDS:
                    p2_summ = OUT_ROOT / exp.feat_tag / _WIN_TAG / eval_kind / "phase2" / f"fold_{f}_seed{s}" / "summary.json"
                    c = _final_c(p2_summ)
                    if c is None:
                        continue
                    vals.append(c)
                    try:
                        j = json.load(open(p2_summ))
                        eps.append(int(j.get("best_epoch", j.get("final_epoch", -1))))
                    except Exception:
                        eps.append(-1)
            rows.append((exp.feat_tag, vals, eps))
        lines.append(f"{'排名':>4s}  {'实验 feat_tag':<64s}  {'N':>3s}   {'mean±std':<16s}  vsLUAD(0.712)")
        ranked = sorted(rows, key=lambda r: (st.mean(r[1]) if r[1] else -1.0), reverse=True)
        luad_ref = 0.712
        for rank, (tag, vals, eps) in enumerate(ranked, 1):
            if not vals:
                lines.append(f"{rank:>4d}  {tag:<64s}  {'0':>3s}   {'未完成':<16s}")
                continue
            m = st.mean(vals)
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            gain_pp = (m - luad_ref) * 100.0
            med_ep = st.median(eps) if eps else float("nan")
            lines.append(f"{rank:>4d}  {tag:<64s}  {len(vals):>3d}   {m:.4f}±{sd:.4f}    {gain_pp:+.2f} pp   med_ep={med_ep:.0f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["phase1", "phase2", "all"], default="all")
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--only-fold", type=int, default=None, help="单折调试（只跑 fold=N）")
    ap.add_argument("--only-seed", type=int, default=None, help="单 seed 调试")
    ap.add_argument("--only-eval", default=None, choices=EVAL_KINDS)
    ap.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES（默认 0 单卡）")
    args = ap.parse_args()

    plans: list[RunPlan] = []
    exps = EXPERIMENTS
    eval_kinds = [args.only_eval] if args.only_eval else EVAL_KINDS
    folds = [args.only_fold] if args.only_fold is not None else list(range(N_FOLDS))
    seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    for exp in exps:
        if args.only in ("phase1", "all"):
            for f in folds:
                for s in seeds:
                    plans.append(build_phase1_plan(exp, f, s, args.cuda))
        if args.only in ("phase2", "all"):
            for f in folds:
                for s in seeds:
                    for ek in eval_kinds:
                        plans.append(build_phase2_plan(exp, f, s, ek, args.cuda))

    if args.start_from:
        idx = next((i for i, p in enumerate(plans) if args.start_from in p.tag), None)
        if idx is None:
            raise SystemExit(f"start-from {args.start_from} 不在计划中：\n" + "\n  ".join(p.tag for p in plans))
        plans = plans[idx:]

    print(f"[INFO] BRCA 计划 {len(plans)} runs  ({len(exps)} exp × folds={folds} × seeds={seeds} × eval_kinds={len(eval_kinds)})")
    print(f"[INFO]  PHASE1：{sum(1 for p in plans if p.tag.startswith('[P1]'))} 次（纯未删失）")
    print(f"[INFO]  PHASE2：{sum(1 for p in plans if p.tag.startswith('[P2]'))} 次（窗口 Cox）")
    print(f"[INFO] _WIN_TAG = {_WIN_TAG}   CUDA={args.cuda}")
    print(f"[INFO] split_root = {SPLITS_ROOT}")
    print(f"[INFO] out_root = {OUT_ROOT}")

    if args.only_report:
        print(summary_report())
        return

    t0 = time.time()
    rcs: list[int] = []
    for plan in plans:
        rcs.append(run_one(plan, skip_existing=not args.no_skip))
    dt = time.time() - t0
    print("\n" + summary_report())
    print(f"\n总耗时 = {dt/3600:.2f} h （{dt/60:.1f} min）  失败 runs = {sum(1 for r in rcs if r != 0)}")


if __name__ == "__main__":
    main()
