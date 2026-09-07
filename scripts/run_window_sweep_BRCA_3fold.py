#!/usr/bin/env python3
"""BRCA Window Sweep（3折×3seeds×多窗口，P1 best.pt 全局共享）。

核心策略：
  1. 参数搜索阶段只用 3 折（folds=[0,1,2]）× 3 seeds = 9 P1 runs，best.pt 共享给所有窗口的 P2
  2. P2 窗口 sweep：候选池 =  LUAD sweep top4 + 1 对照（0.59-0.65 退化款）= 5 组；
     每组 P2 只跑 eval_with_censored 快速出排名
  3. 每窗口输出 Δ = P2 mean − P1 mean，Δ < 0 标记「DEGRADED（课程学习退化）」
  4. 冠军窗口 = 非退化集合中 mean 最高（若全退化则选退化最小的，并打印 WARNING）
  5. 调度器自带 `--only-report`，随时查当前扫到的排名（不启动训练）

候选池窗口来自LUAD 9 组 sweep（表头：排名 窗口 L-U mean±std ep_mode）：
  #1 0.59-0.68  mean=0.6459 (LUAD 冠军)
  #2 0.59-0.72  mean=0.6366
  #3 0.50-0.65  mean=0.6221
  #4 0.55-0.72  mean=0.6209
  —— 对照（LUAD sweep 第7，退化款） ——
  #7 0.59-0.65  mean=0.5809
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
OUT_ROOT = PROJECT_ROOT / "outputs" / "censored_stage_survival_window_sweep_BRCA_3fold"

WINDOW_K = 50.0
WINDOW_A = 0.99
WINDOW_EPS = 1e-3
WINDOW_METRIC = "val_c_index_ema"


@dataclass(frozen=True)
class WindowSpec:
    lower: float
    upper: float
    note: str

    @property
    def tag(self) -> str:
        return f"win{self.lower:.2f}-{self.upper:.2f}_A{WINDOW_A:.2f}_K{WINDOW_K:.0f}"


WINDOW_CANDIDATES: list[WindowSpec] = [
    WindowSpec(0.59, 0.68, "LUAD sweep #1 mean=0.6459"),
    WindowSpec(0.59, 0.72, "LUAD sweep #2 mean=0.6366"),
    WindowSpec(0.50, 0.65, "LUAD sweep #3 mean=0.6221"),
    WindowSpec(0.55, 0.72, "LUAD sweep #4 mean=0.6209"),
    WindowSpec(0.59, 0.65, "LUAD sweep #7 mean=0.5809 (对照·退化款)"),
]

SWEEP_FOLDS = [0, 1, 2]
SWEEP_SEEDS = [0, 1, 2]
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50
EVAL_KIND_SWEEP = "eval_with_censored"
GTF_PATH = "/root/autodl-tmp/gencode.v22.annotation.gtf.gz"
PHASE1_DIRNAME = "_shared_phase1"


@dataclass(frozen=True)
class ExpSpec:
    feat_tag: str
    wsi_feature_source: str
    wsi_feature_source_aliases: str
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
    cuda: str


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
    out_dir = OUT_ROOT / exp.feat_tag / PHASE1_DIRNAME / f"fold_{fold}_seed{seed}"
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


def build_phase2_plan(exp: ExpSpec, fold: int, seed: int, ws: WindowSpec, cuda: str) -> RunPlan:
    split_dir = PHASE2_ROOT / f"fold_{fold}" / EVAL_KIND_SWEEP
    pretrain = OUT_ROOT / exp.feat_tag / PHASE1_DIRNAME / f"fold_{fold}_seed{seed}" / "best.pt"
    out_dir = OUT_ROOT / exp.feat_tag / ws.tag / EVAL_KIND_SWEEP / f"fold_{fold}_seed{seed}"
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
        "--loss_window_lower", str(ws.lower),
        "--loss_window_upper", str(ws.upper),
        "--loss_window_k", str(WINDOW_K),
        "--loss_window_A", str(WINDOW_A),
        "--loss_window_eps", str(WINDOW_EPS),
        "--loss_window_metric", WINDOW_METRIC,
    ]
    return RunPlan(tag=f"[P2]{exp.feat_tag[:18]}.._f{fold}_s{seed}_{ws.tag}", cmd=cmd, out_dir=out_dir, log_file=log_file, cuda=cuda)


def _marker_exists(out_dir: Path) -> bool:
    ok_ckpt = (out_dir / "final.pt").exists() or (out_dir / "best.pt").exists()
    ok_summ = (out_dir / "summary.json").exists()
    return bool(ok_ckpt and ok_summ)


def run_one(plan: RunPlan, skip_existing: bool = True) -> int:
    if skip_existing and _marker_exists(plan.out_dir):
        print(f"\n[SKIP] {plan.tag}  summary+ckpt 已存在 → 跳过", flush=True)
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


def _best_epoch(summary_json: Path) -> int:
    if not summary_json.exists():
        return -1
    try:
        j = json.load(open(summary_json))
    except Exception:
        return -1
    for k in ("best_epoch", "final_epoch"):
        if k in j:
            try:
                return int(j[k])
            except Exception:
                pass
    return -1


def _p1_vals(exp: ExpSpec) -> list[float]:
    vals: list[float] = []
    for f in SWEEP_FOLDS:
        for s in SWEEP_SEEDS:
            summ = OUT_ROOT / exp.feat_tag / PHASE1_DIRNAME / f"fold_{f}_seed{s}" / "summary.json"
            c = _final_c(summ)
            if c is not None:
                vals.append(c)
    return vals


def summary_report() -> str:
    lines: list[str] = []
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    lines.append("=" * 130)
    lines.append(f"BRCA WINDOW SWEEP 3-FOLD 排名报告  ({t0})   folds={SWEEP_FOLDS}  seeds={SWEEP_SEEDS}  eval={EVAL_KIND_SWEEP}")
    lines.append(f"Experiment: {EXPERIMENTS[0].feat_tag}  (aliases={EXPERIMENTS[0].wsi_feature_source_aliases})")
    lines.append("")

    exp = EXPERIMENTS[0]
    p1_vals = _p1_vals(exp)
    if p1_vals:
        p1_m, p1_sd = st.mean(p1_vals), (st.pstdev(p1_vals) if len(p1_vals) > 1 else 0.0)
        lines.append(f"[PHASE1 SHARED]  N={len(p1_vals):>2d}/9   mean±std = {p1_m:.4f}±{p1_sd:.4f}   (基线 Δ=0)")
    else:
        p1_m, p1_sd = float("nan"), float("nan")
        lines.append("[PHASE1 SHARED]  暂无完成 runs（等待 P1 跑完）")
    lines.append("")

    rows: list[tuple[WindowSpec, list[float], list[int]]] = []
    for ws in WINDOW_CANDIDATES:
        vals: list[float] = []
        eps: list[int] = []
        for f in SWEEP_FOLDS:
            for s in SWEEP_SEEDS:
                summ = OUT_ROOT / exp.feat_tag / ws.tag / EVAL_KIND_SWEEP / f"fold_{f}_seed{s}" / "summary.json"
                c = _final_c(summ)
                if c is not None:
                    vals.append(c)
                    eps.append(_best_epoch(summ))
        rows.append((ws, vals, eps))

    lines.append(f"{'排名':>4s}  {'窗口 L-U':<14s}  {'N':>3s}  {'mean±std':<16s}  {'Δ=P2-P1(pp)':>12s}  {'状态':<12s}  ep_mode  备注")
    valid: list[tuple[float, float, bool, int, WindowSpec, list[float], list[int]]] = []
    for (ws, vals, eps) in rows:
        if not vals:
            continue
        m = st.mean(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        delta_pp = (m - p1_m) * 100.0 if p1_m == p1_m else float("nan")
        degraded = delta_pp < 0 if delta_pp == delta_pp else False
        med_ep = int(st.median(eps)) if eps else -1
        valid.append((-m, sd, degraded, med_ep, ws, vals, eps))
    valid.sort()
    rank = 0
    for (_, sd, degraded, med_ep, ws, vals, eps) in valid:
        rank += 1
        m = -_
        m = st.mean(vals)
        delta_pp = (m - p1_m) * 100.0 if p1_m == p1_m else float("nan")
        state = "❌DEGRADED" if degraded else "✅OK"
        delta_str = f"{delta_pp:+.2f}" if delta_pp == delta_pp else "N/A"
        lines.append(
            f"{rank:>4d}  {ws.lower:.2f}-{ws.upper:.2f}      {len(vals):>3d}  {m:.4f}±{sd:.4f}    {delta_str:>10s}      {state:<12s}  {med_ep:>3d}     {ws.note}"
        )
    if len(valid) < len(WINDOW_CANDIDATES):
        for (ws, vals, eps) in rows:
            if vals:
                continue
            lines.append(f"  --  {ws.lower:.2f}-{ws.upper:.2f}        0  未完成                                  -1     {ws.note}")
    lines.append("")
    if valid:
        non_deg = [v for v in valid if not v[2]]
        if non_deg:
            best = non_deg[0]
            best_m = st.mean(best[5])
            lines.append(f"🏆 推荐冠军窗口（非退化 top）：{best[4].lower:.2f}-{best[4].upper:.2f}   N={len(best[5])}  mean={best_m:.4f}  备注：{best[4].note}")
            lines.append(f"   → 下一步：用 5 折官方验证（folds=[0,1,2,3,4] × seeds={SWEEP_SEEDS} × 2 eval_kind）跑全量 45 runs 出最终报告")
        else:
            best = valid[0]
            best_m = st.mean(best[5])
            best_delta = (best_m - p1_m) * 100.0 if p1_m == p1_m else float("nan")
            lines.append(f"⚠️  WARNING：所有候选窗口全部退化！最佳退化最少者 {best[4].lower:.2f}-{best[4].upper:.2f}   mean={best_m:.4f}  Δ={best_delta:+.2f} pp")
            lines.append("   → 建议：扩大候选窗口 upper (0.75/0.80) 或提高 lower (0.62) 重扫；或检查 best_epoch/selection_metric 是否触发 early-stop 前窗口已激活")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["phase1", "phase2", "all"], default="all")
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--only-window", default=None, help="只跑特定窗口，格式 '0.59-0.68'")
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES（默认 0 单卡）")
    args = ap.parse_args()

    windows = WINDOW_CANDIDATES
    if args.only_window:
        try:
            lstr, ustr = args.only_window.split("-")
            l_, u_ = float(lstr), float(ustr)
        except Exception:
            raise SystemExit(f"--only-window 格式应为 'L-U'（例 0.59-0.68），收到 {args.only_window}")
        windows = [w for w in windows if abs(w.lower - l_) < 1e-6 and abs(w.upper - u_) < 1e-6]
        if not windows:
            raise SystemExit(f"--only-window={args.only_window} 不在候选池：{[(w.lower,w.upper) for w in WINDOW_CANDIDATES]}")

    folds = [args.only_fold] if args.only_fold is not None else SWEEP_FOLDS
    seeds = [args.only_seed] if args.only_seed is not None else SWEEP_SEEDS

    plans: list[RunPlan] = []
    exps = EXPERIMENTS
    for exp in exps:
        if args.only in ("phase1", "all"):
            for f in folds:
                for s in seeds:
                    plans.append(build_phase1_plan(exp, f, s, args.cuda))
        if args.only in ("phase2", "all"):
            for ws in windows:
                for f in folds:
                    for s in seeds:
                        plans.append(build_phase2_plan(exp, f, s, ws, args.cuda))

    if args.start_from:
        idx = next((i for i, p in enumerate(plans) if args.start_from in p.tag), None)
        if idx is None:
            raise SystemExit(f"start-from {args.start_from} 不在计划中：\n" + "\n  ".join(p.tag[:80] for p in plans))
        plans = plans[idx:]

    n_p1 = sum(1 for p in plans if p.tag.startswith("[P1]"))
    n_p2 = sum(1 for p in plans if p.tag.startswith("[P2]"))
    print(f"[INFO] BRCA Window Sweep 计划 {len(plans)} runs  ({len(exps)} exp × folds={folds} × seeds={seeds})")
    print(f"[INFO]  PHASE1 (shared)：{n_p1} 次（纯未删失，best.pt 被所有窗口复用）")
    print(f"[INFO]  PHASE2 (sweep) ：{n_p2} 次（{len(windows)} 窗口 × {len(folds)*len(seeds)} fold-seeds）")
    print(f"[INFO]  候选窗口池 ({len(windows)}): {[(w.lower,w.upper,w.note[:22]) for w in windows]}")
    print(f"[INFO]  split_root = {SPLITS_ROOT}")
    print(f"[INFO]  out_root = {OUT_ROOT}   CUDA={args.cuda}")

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
