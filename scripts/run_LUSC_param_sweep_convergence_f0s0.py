#!/usr/bin/env python3
"""LUSC 参数收敛 sweep（单 fold_0 + seed_0，含 P1 自动训练，~30min + 29 P2）。
对齐 LUAD sweep 格式（A9 组 + window_grid 20 组），3 癌种横向可比。
 #1 窗口收敛 + #2 A 收敛 + #6 激活量化 → 直接复用 LUAD sweep 的报告解析/激活解析。
P1 不存在时自动先训 P1 fold0_seed0（训完以后续 rerun 可复用，best.pt/summary.json 自动写）。
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
COHORT = "LUSC"
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits" / f"censored_stage_protocol_{COHORT}"
PHASE1_ROOT = SPLITS_ROOT / "phase1_outer5"
PHASE2_ROOT = SPLITS_ROOT / "phase2_independent5"
OUT_ROOT = PROJECT_ROOT / "outputs" / f"censored_stage_survival_{COHORT}_param_sweep_f0s0"

PRETRAIN_P1_BEST = OUT_ROOT / "_shared_phase1" / "fold_0_seed0" / "best.pt"
P1_SUMMARY_JSON = PRETRAIN_P1_BEST.parent / "summary.json"

FIXED_WINDOW_FOR_A = (0.59, 0.68)
WINDOW_K = 50.0
WINDOW_EPS = 1e-3
WINDOW_METRIC = "val_c_index_ema"

SWEEP_FOLD = 0
SWEEP_SEED = 0
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50
EVAL_KIND = "eval_with_censored"
GTF_PATH = "/root/autodl-tmp/gencode.v22.annotation.gtf.gz"


@dataclass(frozen=True)
class WindowSpec:
    lower: float
    upper: float
    A: float = 0.99
    note: str = ""

    @property
    def win_tag(self) -> str:
        return f"win{self.lower:.2f}-{self.upper:.2f}_A{self.A:.3f}_K{WINDOW_K:.0f}"


def _build_A_candidates() -> list[WindowSpec]:
    L, U = FIXED_WINDOW_FOR_A
    As = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
    return [WindowSpec(L, U, A, f"A={A:.3f}") for A in As]


def _build_window_grid() -> list[WindowSpec]:
    Ls = [0.50, 0.53, 0.56, 0.59, 0.62]
    Us = [0.65, 0.68, 0.72, 0.80]
    out: list[WindowSpec] = []
    for L in Ls:
        for U in Us:
            if U - L < 0.03:
                continue
            out.append(WindowSpec(L, U, 0.99, f"L={L:.2f} U={U:.2f}"))
    return out


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
        "--cohort_hint", COHORT,
    ]


def build_phase1_plan(exp: ExpSpec, cuda: str) -> RunPlan:
    split_dir = PHASE1_ROOT / f"fold_{SWEEP_FOLD}"
    out_dir = PRETRAIN_P1_BEST.parent
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *build_base_args(exp),
        "--split_dir", str(split_dir),
        "--stage", "1",
        "--epochs", str(EPOCHS_PHASE1),
        "--seed", str(SWEEP_SEED),
        "--out_dir", str(out_dir),
    ]
    return RunPlan(tag=f"[P1]{COHORT}_f{SWEEP_FOLD}s{SWEEP_SEED}", cmd=cmd, out_dir=out_dir, log_file=log_file, cuda=cuda)


def build_phase2_plan(exp: ExpSpec, ws: WindowSpec, sweep_type_dir: str, cuda: str) -> RunPlan:
    split_dir = PHASE2_ROOT / f"fold_{SWEEP_FOLD}" / EVAL_KIND
    out_dir = OUT_ROOT / sweep_type_dir / exp.feat_tag / ws.win_tag / f"fold_{SWEEP_FOLD}_seed{SWEEP_SEED}"
    log_file = out_dir / "run.log"
    cmd = [
        str(PYTHON), str(SCRIPT),
        *build_base_args(exp),
        "--split_dir", str(split_dir),
        "--stage", "2",
        "--pretrain_checkpoint", str(PRETRAIN_P1_BEST),
        "--epochs", str(EPOCHS_PHASE2),
        "--seed", str(SWEEP_SEED),
        "--out_dir", str(out_dir),
        "--loss_window_lower", str(ws.lower),
        "--loss_window_upper", str(ws.upper),
        "--loss_window_k", str(WINDOW_K),
        "--loss_window_A", str(ws.A),
        "--loss_window_eps", str(WINDOW_EPS),
        "--loss_window_metric", WINDOW_METRIC,
    ]
    return RunPlan(tag=f"[P2]{sweep_type_dir[:3]}_{ws.win_tag}", cmd=cmd, out_dir=out_dir, log_file=log_file, cuda=cuda)


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


def _parse_activation_epoch(run_log: Path) -> tuple[int | None, float | None, float | None]:
    if not run_log.exists():
        return None, None, None
    try:
        txt = run_log.read_text(errors="ignore")
    except Exception:
        return None, None, None
    act_ep = None
    for line in txt.splitlines():
        if "epoch=" not in line or "w_now=" not in line:
            continue
        try:
            ep = int(line.split("epoch=")[1].split()[0])
            w = line.split("w_now=")[1].split()[0]
            wf = float(w)
            if wf < 0.1 and act_ep is None:
                act_ep = ep
                break
        except Exception:
            continue
    if act_ep is None:
        return None, None, None
    test_c_by_ep: dict[int, float] = {}
    for line in txt.splitlines():
        if "epoch=" not in line or "test_c=" not in line:
            continue
        try:
            ep = int(line.split("epoch=")[1].split()[0])
            tc = float(line.split("test_c=")[1].split()[0])
            test_c_by_ep[ep] = tc
        except Exception:
            continue
    if not test_c_by_ep:
        return act_ep, None, None
    before = [v for e, v in test_c_by_ep.items() if e < act_ep]
    after = [v for e, v in test_c_by_ep.items() if e >= act_ep]
    bm = max(before) if before else None
    am = max(after) if after else None
    return act_ep, bm, am


def _p1_c() -> float:
    c = _final_c(P1_SUMMARY_JSON)
    return c if c is not None else float("nan")


def _find_inflection_point_1d(xs_sorted: list[float], ys_sorted: list[float]) -> float | None:
    if len(ys_sorted) < 4:
        return None
    d1 = [ys_sorted[i+1] - ys_sorted[i] for i in range(len(ys_sorted)-1)]
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)]
    abs_d2 = [abs(d) for d in d2]
    i = max(range(len(abs_d2)), key=lambda k: abs_d2[k])
    return xs_sorted[i+1]


def summary_report_A() -> str:
    lines: list[str] = []
    p1_c = _p1_c()
    lines.append(f"\n{'='*110}")
    lines.append(f"[{COHORT} A 收敛]  fold=0 seed=0   窗口={FIXED_WINDOW_FOR_A}  P1_baseline={p1_c:.4f}")
    cands = _build_A_candidates()
    exp = EXPERIMENTS[0]
    rows: list[tuple[float, WindowSpec, float | None, int, float | None, float | None, float | None]] = []
    for ws in cands:
        summ = OUT_ROOT / "sweep_A" / exp.feat_tag / ws.win_tag / f"fold_{SWEEP_FOLD}_seed{SWEEP_SEED}" / "summary.json"
        run_log = summ.parent / "run.log"
        c = _final_c(summ)
        ep = _best_epoch(summ)
        act_ep, tc_b, tc_a = _parse_activation_epoch(run_log)
        rows.append((ws.A, ws, c, ep, act_ep, tc_b, tc_a))
    rows.sort()
    lines.append(f"{'A':>6s}  {'C-index':>8s}  {'Δpp(P2-P1)':>10s}  {'状态':>10s}  best_ep  act_ep  C_pre  C_post  Δ_post_pre(pp)  备注")
    valid: list[tuple[float, float]] = []
    for (A, ws, c, ep, act_ep, tc_b, tc_a) in rows:
        if c is None:
            lines.append(f"{A:6.3f}  {'未完成':>8s}  {'':>10s}  {'':>10s}  {-1:>7d}  {(-1 if act_ep is None else act_ep):>6d}  {'':>5s}  {'':>6s}  {'':>12s}  {ws.note}")
            continue
        delta_pp = (c - p1_c) * 100.0 if p1_c == p1_c else float("nan")
        state = "✅OK" if (delta_pp == delta_pp and delta_pp >= 0) else "❌DEG"
        post_pre_pp = f"{(tc_a-tc_b)*100:+.2f}" if (tc_b is not None and tc_a is not None) else "N/A"
        tc_b_s = f"{tc_b:.3f}" if tc_b is not None else "N/A"
        tc_a_s = f"{tc_a:.3f}" if tc_a is not None else "N/A"
        delta_str = f"{delta_pp:+.2f}" if delta_pp == delta_pp else "N/A"
        lines.append(f"{A:6.3f}  {c:8.4f}  {delta_str:>10s}  {state:>10s}  {ep:>7d}  {(-1 if act_ep is None else act_ep):>6d}  {tc_b_s:>5s}  {tc_a_s:>6s}  {post_pre_pp:>12s}  {ws.note}")
        valid.append((A, c))
    if valid:
        xs = [x for x, _ in valid]
        ys = [y for _, y in valid]
        infl_A = _find_inflection_point_1d(xs, ys)
        best = max(valid, key=lambda t: t[1])
        lines.append(f"\n🏆 A 最佳 (C 最大)：A={best[0]:.3f}  C={best[1]:.4f}   Δ={(best[1]-p1_c)*100:+.2f}pp vs P1")
        if infl_A is not None:
            lines.append(f"📉 A 收敛拐点 (二阶差分最大)：A≈{infl_A:.3f} （建议论文取 A≈{infl_A:.3f}~{best[0]:.3f} 区间）")
    return "\n".join(lines)


def summary_report_window() -> str:
    lines: list[str] = []
    p1_c = _p1_c()
    lines.append(f"\n{'='*130}")
    lines.append(f"[{COHORT} 窗口收敛]  fold=0 seed=0   A=0.99   P1_baseline={p1_c:.4f}")
    cands = _build_window_grid()
    exp = EXPERIMENTS[0]
    rows: list[tuple[float, float, WindowSpec, float | None, int, float | None, float | None, float | None]] = []
    for ws in cands:
        summ = OUT_ROOT / "sweep_window" / exp.feat_tag / ws.win_tag / f"fold_{SWEEP_FOLD}_seed{SWEEP_SEED}" / "summary.json"
        run_log = summ.parent / "run.log"
        c = _final_c(summ)
        ep = _best_epoch(summ)
        act_ep, tc_b, tc_a = _parse_activation_epoch(run_log)
        rows.append((ws.lower, ws.upper, ws, c, ep, act_ep, tc_b, tc_a))
    Ls = sorted(set(r[0] for r in rows))
    Us = sorted(set(r[1] for r in rows))
    lines.append("\n[C-index 热图（行=lower L，列=upper U）]：")
    header = f"{'L\\U':<6s}  " + "  ".join(f"{U:.2f}" for U in Us)
    lines.append(header)
    c_by_LU: dict[tuple[float, float], float | None] = {(r[0], r[1]): r[3] for r in rows}
    for L in Ls:
        cells = []
        for U in Us:
            c = c_by_LU.get((L, U), None)
            cells.append(f"{c:6.4f}" if c is not None else "  --  ")
        lines.append(f"{L:<6.2f}  " + "  ".join(cells))
    rows_s: list[tuple[float, float, float, WindowSpec, int, float | None, float | None, float | None]] = []
    for (L, U, ws, c, ep, act_ep, tc_b, tc_a) in rows:
        if c is None:
            continue
        rows_s.append((c, L, U, ws, ep, act_ep, tc_b, tc_a))
    rows_s.sort(reverse=True)
    lines.append(f"\n{'排名':>3s}  {'L-U':<12s}  C-index  Δpp(P2-P1)  {'状态':>8s}  best_ep  act_ep  C_pre  C_post  Δ_post_pre(pp)")
    for rank, (c, L, U, ws, ep, act_ep, tc_b, tc_a) in enumerate(rows_s, 1):
        delta_pp = (c - p1_c) * 100.0 if p1_c == p1_c else float("nan")
        state = "✅OK" if (delta_pp == delta_pp and delta_pp >= 0) else "❌DEG"
        post_pre_pp = f"{(tc_a-tc_b)*100:+.2f}" if (tc_b is not None and tc_a is not None) else "N/A"
        tc_b_s = f"{tc_b:.3f}" if tc_b is not None else "N/A"
        tc_a_s = f"{tc_a:.3f}" if tc_a is not None else "N/A"
        delta_str = f"{delta_pp:+.2f}" if delta_pp == delta_pp else "N/A"
        lines.append(f"{rank:>3d}  {L:.2f}-{U:.2f}     {c:.4f}  {delta_str:>10s}  {state:>8s}  {ep:>7d}  {(-1 if act_ep is None else act_ep):>6d}  {tc_b_s:>5s}  {tc_a_s:>6s}  {post_pre_pp:>12s}")
    if rows_s:
        best = rows_s[0]
        lines.append(f"\n🏆 窗口最佳：L={best[1]:.2f}-U={best[2]:.2f}   C={best[0]:.4f}   Δ={(best[0]-p1_c)*100:+.2f}pp vs P1")
        max_U = {L: max([(c_by_LU.get((L, U)) or -1.0, U) for U in Us]) for L in Ls}
        lines.append("📉 各 L 下最优 U（收敛建议）：" + ", ".join(f"L={L:.2f}→U={max_U[L][1]:.2f}(C={max_U[L][0]:.3f})" for L in Ls if max_U[L][0] > 0))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-type", required=True, choices=["A", "window_grid", "both"])
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    cuda = args.cuda
    exp = EXPERIMENTS[0]

    if not PRETRAIN_P1_BEST.exists():
        if args.only_report:
            print(f"[WARN] P1 best.pt 不存在（{PRETRAIN_P1_BEST}），P1_baseline 无法加载（only-report 仍继续）")
        else:
            p1_plan = build_phase1_plan(exp, cuda)
            print(f"[INFO] {COHORT} P1 fold0_seed0 best.pt 不存在 → 先训 P1（~30min）", flush=True)
            rc = run_one(p1_plan, skip_existing=not args.no_skip)
            if rc != 0:
                raise SystemExit(f"[FATAL] {COHORT} P1 训练失败 rc={rc} → 无法续跑 P2 sweep")
            if not PRETRAIN_P1_BEST.exists():
                raise SystemExit(f"[FATAL] {COHORT} P1 训完仍无 best.pt → 检查 {PRETRAIN_P1_BEST.parent}")
            print(f"[OK] {COHORT} P1 完成，best C-index = {_p1_c():.4f}", flush=True)

    plans: list[RunPlan] = []
    sweeps_in_order: list[tuple[str, list[WindowSpec]]] = []
    if args.sweep_type in ("A", "both"):
        sweeps_in_order.append(("sweep_A", _build_A_candidates()))
    if args.sweep_type in ("window_grid", "both"):
        sweeps_in_order.append(("sweep_window", _build_window_grid()))
    for (st_dir, specs) in sweeps_in_order:
        for ws in specs:
            plans.append(build_phase2_plan(exp, ws, st_dir, cuda))

    if args.start_from:
        idx = next((i for i, p in enumerate(plans) if args.start_from in p.tag), None)
        if idx is None:
            raise SystemExit(f"start-from {args.start_from} 不在计划：\n" + "\n  ".join(p.tag[:80] for p in plans))
        plans = plans[idx:]

    print(f"[INFO] {COHORT} 参数收敛 sweep 计划 {len(plans)} runs   (fold={SWEEP_FOLD} seed={SWEEP_SEED})")
    print(f"[INFO]   P1 best.pt（存在={PRETRAIN_P1_BEST.exists()}）：{PRETRAIN_P1_BEST}")
    for (st_dir, specs) in sweeps_in_order:
        print(f"[INFO]   sweep-type={st_dir}: {len(specs)} 组参数")
    print(f"[INFO]   out_root = {OUT_ROOT}   CUDA={cuda}")

    if args.only_report:
        for (st_dir, _) in sweeps_in_order:
            if st_dir == "sweep_A":
                print(summary_report_A())
            else:
                print(summary_report_window())
        return

    t0 = time.time()
    rcs: list[int] = []
    for plan in plans:
        rcs.append(run_one(plan, skip_existing=not args.no_skip))
    dt = time.time() - t0
    for (st_dir, _) in sweeps_in_order:
        if st_dir == "sweep_A":
            print(summary_report_A())
        else:
            print(summary_report_window())
    print(f"\n总耗时 = {dt/3600:.2f} h （{dt/60:.1f} min）  失败 runs = {sum(1 for r in rcs if r != 0)}")


if __name__ == "__main__":
    main()
