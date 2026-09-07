#!/usr/bin/env python3
"""消融分析统一扫描器（覆盖用户 7 天计划的 #3/#4/#5/#6）。

单命令 `python analyze_ablation_aggregate.py` 扫 `R2wsp/outputs/` 全部 summary.json，输出：
  1. ablation_matrix.csv  —— 4 癌种 × 3 WSI 编码器 × 2 RNA 消融 × 2 eval_kind 的 N / mean±std / median best_epoch
  2. censoring_reversal.csv —— 各癌种 Δ = eval_uncensored_only − eval_with_censored（Δ<0 → 「删失数据反转」计数 + 量化幅度）
  3. activation_window_quant.csv —— 窗口激活 epoch / C_pre / C_post / Δ_post_pre 分布（需要 run.log 存在）
  4. aggregate.json —— 结构化数据，给画图脚本直接 load
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs"
SAVE_DIR = OUT_ROOT / "_analysis_aggregate_7day"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS_ALL = ["LUAD", "LUSC", "BRCA", "BLCA"]
EVAL_KINDS = ["eval_with_censored", "eval_uncensored_only"]
WSI_TAGS = {
    "UNI": "UNI1024_tilefix256",
    "DINO": "DINOv2_ViTL_tilefix256",
    "PLIP": "PLIP",
}
RNA_TAGS = {
    "omics": "hallmark50_omics",
    "vec":   "hallmark50_vec",
}
P1_WSI_ALIASES = {  # 从路径 feat_tag 反推 (wsi_model, rna_mode)
    "wsi=UNI1024_tilefix256_rna=hallmark50_omics": ("UNI", "omics"),
    "wsi=UNI1024_tilefix256_rna=hallmark50_vec": ("UNI", "vec"),
    "wsi=DINOv2_ViTL_tilefix256_rna=hallmark50_omics": ("DINO", "omics"),
    "wsi=DINOv2_ViTL_tilefix256_rna=hallmark50_vec": ("DINO", "vec"),
    "wsi=PLIP_rna=hallmark50_omics": ("PLIP", "omics"),
    "wsi=PLIP_rna=hallmark50_vec": ("PLIP", "vec"),
    # rna_branch_baseline 老格式
    "branch=omics": (None, "omics"),
    "branch=vec":   (None, "vec"),
    "branch=dense_vec": (None, "vec"),
    "branch=dense_omics": (None, "omics"),
}

WIN_RE = re.compile(r"win([0-9.]+)-([0-9.]+)_A([0-9.]+)_K([0-9]+)")
FOLD_RE = re.compile(r"fold_([0-9]+)_seed([0-9]+)")


@dataclass
class Row:
    cohort: str | None = None
    wsi_model: str | None = None
    rna_mode: str | None = None
    eval_kind: str | None = None
    window_l: float | None = None
    window_u: float | None = None
    window_A: float | None = None
    fold: int | None = None
    seed: int | None = None
    stage: int | None = None
    test_c: float | None = None
    best_epoch: int | None = None
    act_epoch: int | None = None
    test_c_pre_act: float | None = None
    test_c_post_act: float | None = None
    source_dir: str = ""

    @property
    def key_abl(self) -> tuple[str, str, str, str]:
        return (self.cohort or "UNK", self.wsi_model or "UNK", self.rna_mode or "UNK", self.eval_kind or "UNK")


def _final_c(j: dict) -> float | None:
    for k in ("final_test_c_index", "best_test_c_index", "test_c_index_ema", "test_c_index"):
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def _best_epoch(j: dict) -> int | None:
    for k in ("best_epoch", "final_epoch"):
        if k in j:
            try:
                return int(j[k])
            except Exception:
                pass
    return None


def _infer_cohort_from_path(p: Path) -> str | None:
    s = str(p)
    for c in COHORTS_ALL:
        if f"/{c}/" in s or f"_{c}_" in s or s.endswith(f"/{c}") or s.endswith(f"_{c}"):
            return c
    if "censored_stage_survival_full_upgrade" in s and "_BRCA" not in s:
        return "LUAD"
    if "anti_plip" in s.lower() or "base_plip" in s.lower() or "plip_vec" in s.lower():
        return "LUAD"
    if "LUAD" in s:
        return "LUAD"
    return None


def _parse_act_from_log(run_log: Path) -> tuple[int | None, float | None, float | None]:
    if not run_log.exists():
        return None, None, None
    try:
        txt = run_log.read_text(errors="ignore")
    except Exception:
        return None, None, None
    act_ep: int | None = None
    for line in txt.splitlines():
        if "epoch=" not in line or "w_now=" not in line:
            continue
        try:
            ep = int(line.split("epoch=")[1].split()[0])
            w = line.split("w_now=")[1].split()[0]
            if w == "nan":
                continue
            if float(w) < 0.1 and act_ep is None:
                act_ep = ep
                break
        except Exception:
            continue
    if act_ep is None:
        return None, None, None
    tc_by_ep: dict[int, float] = {}
    for line in txt.splitlines():
        if "epoch=" not in line or "test_c=" not in line:
            continue
        try:
            ep = int(line.split("epoch=")[1].split()[0])
            tc = float(line.split("test_c=")[1].split()[0])
            tc_by_ep[ep] = tc
        except Exception:
            continue
    if not tc_by_ep:
        return act_ep, None, None
    pre = [v for e, v in tc_by_ep.items() if e < act_ep]
    post = [v for e, v in tc_by_ep.items() if e >= act_ep]
    b = max(pre) if pre else None
    a = max(post) if post else None
    return act_ep, b, a


def _parse_feat_tag_from_path(p: Path) -> tuple[str | None, str | None]:
    parts = p.parts
    s = str(p)
    for part in parts:
        if part.startswith("wsi=") and "_rna=" in part:
            key = part
            if key in P1_WSI_ALIASES:
                return P1_WSI_ALIASES[key]
            m = re.search(r"wsi=([^_]+(?:_[^=]+)*?)_rna=(.+)$", key)
            if m:
                raw_wsi = m.group(1)
                raw_rna = m.group(2)
                wsi = None
                if "UNI1024" in raw_wsi:
                    wsi = "UNI"
                elif "DINOv2" in raw_wsi or "DINO" in raw_wsi:
                    wsi = "DINO"
                elif "PLIP" in raw_wsi:
                    wsi = "PLIP"
                rna = None
                if "hallmark50_omics" in raw_rna or raw_rna.endswith("_omics") or "/omics" in s:
                    rna = "omics"
                elif "hallmark50_vec" in raw_rna or raw_rna.endswith("_vec") or "/vec" in s:
                    rna = "vec"
                return (wsi, rna)
    for part in parts:
        if part in P1_WSI_ALIASES:
            return P1_WSI_ALIASES[part]
    # anti_plip_vec / base_plip_vec 老格式：wsi_model = PLIP，rna = vec (看名字)
    if "anti_plip" in s.lower() or "base_plip" in s.lower() or "sweep_censored_stage_survival_anti_plip" in s:
        rna = "vec" if "_vec_" in s or "_vec_LUAD" in s else (
            "omics" if "_omics_" in s or "_omics_LUAD" in s else None
        )
        return ("PLIP", rna)
    if "/censored_stage_survival_full_upgrade/" in s and "_BRCA" not in s:
        return ("UNI", "omics")  # LUAD full upgrade 默认 UNI + omics
    return None, None


def _parse_eval_kind_from_path(p: Path) -> str | None:
    for ek in EVAL_KINDS:
        if ek in str(p):
            return ek
    return None


def _parse_window_from_path(p: Path) -> tuple[float | None, float | None, float | None]:
    s = str(p)
    m = WIN_RE.search(s)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _parse_fold_seed_from_path(p: Path) -> tuple[int | None, int | None]:
    s = str(p)
    m = FOLD_RE.search(s)
    if not m:
        # rna_branch_baseline 老格式：k=0/seed=0
        km = re.search(r"k=([0-9]+)/.+seed=([0-9]+)", s)
        if km:
            return int(km.group(1)), int(km.group(2))
        return None, None
    return int(m.group(1)), int(m.group(2))


def collect_rows() -> list[Row]:
    rows: list[Row] = []
    for summary_json in OUT_ROOT.rglob("summary.json"):
        if "_analysis_aggregate_7day" in str(summary_json):
            continue
        try:
            j = json.load(open(summary_json))
        except Exception:
            continue
        tc = _final_c(j)
        if tc is None:
            continue
        r = Row(test_c=tc, source_dir=str(summary_json.parent))
        r.cohort = _infer_cohort_from_path(summary_json) or (
            j.get("cohort") if isinstance(j, dict) else None
        )
        r.wsi_model, r.rna_mode = _parse_feat_tag_from_path(summary_json)
        r.eval_kind = _parse_eval_kind_from_path(summary_json)
        r.window_l, r.window_u, r.window_A = _parse_window_from_path(summary_json)
        r.fold, r.seed = _parse_fold_seed_from_path(summary_json)
        r.best_epoch = _best_epoch(j)
        if "stage" in j:
            try:
                r.stage = int(j["stage"])
            except Exception:
                r.stage = 2 if ("phase2" in str(summary_json)) else (1 if "phase1" in str(summary_json) else None)
        else:
            r.stage = 2 if ("phase2" in str(summary_json)) else (1 if "phase1" in str(summary_json) else None)
        run_log = summary_json.parent / "run.log"
        r.act_epoch, r.test_c_pre_act, r.test_c_post_act = _parse_act_from_log(run_log)
        rows.append(r)
    return rows


def main() -> None:
    rows = collect_rows()
    print(f"[INFO] 扫描到 {len(rows)} 条 summary.json 记录（含未完成的 P1/P2，无 test_c 已过滤）")

    # === Output 1: ablation_matrix.csv ===
    key_vals: dict[tuple[str, str, str, str], list[Row]] = {}
    for r in rows:
        if r.stage and r.stage != 2:
            continue
        k = r.key_abl
        if k[0] == "UNK":
            continue
        key_vals.setdefault(k, []).append(r)

    f_csv = SAVE_DIR / "ablation_matrix.csv"
    with f_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "wsi_model", "rna_mode", "eval_kind", "N_runs", "mean_c", "std_c",
                    "median_c", "min_c", "max_c", "median_best_epoch", "coh_abbr"])
        all_keys = sorted(key_vals.keys())
        for k in all_keys:
            rs = key_vals[k]
            cs = [r.test_c for r in rs if r.test_c is not None]
            eps = [r.best_epoch for r in rs if r.best_epoch is not None]
            if not cs:
                continue
            cohort, wsi, rna, ek = k
            w.writerow([
                cohort, wsi or "", rna or "", ek, len(cs),
                f"{st.mean(cs):.6f}", f"{(st.pstdev(cs) if len(cs)>1 else 0.0):.6f}",
                f"{st.median(cs):.6f}", f"{min(cs):.6f}", f"{max(cs):.6f}",
                int(st.median(eps)) if eps else -1,
                f"{cohort}_{wsi}_{rna}_{ek[:4]}",
            ])
    print(f"[DONE] {f_csv}（{len(all_keys)} 个 (cohort×wsi×rna×eval) 组合）")

    # === Output 2: censoring_reversal.csv（用户 #4：删失造成的数据反转 = Δ=uncens - with_cens）===
    reversal_csv = SAVE_DIR / "censoring_reversal.csv"
    by_cohort_wsi_rna: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for r in rows:
        if r.stage and r.stage != 2:
            continue
        if r.cohort == "UNK":
            continue
        if not r.eval_kind:
            continue
        key = (r.cohort or "UNK", r.wsi_model or "UNK", r.rna_mode or "UNK")
        by_cohort_wsi_rna.setdefault(key, {"eval_with_censored": [], "eval_uncensored_only": []})
        if r.eval_kind in by_cohort_wsi_rna[key] and r.test_c is not None:
            by_cohort_wsi_rna[key][r.eval_kind].append(r.test_c)
    with reversal_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "wsi_model", "rna_mode",
                    "N_with_cens", "mean_with_cens",
                    "N_uncens_only", "mean_uncens_only",
                    "Δpp (uncens - with_cens)",  # Δ < 0 = 反转（删失加进去反而 C 下降）
                    "N_runs_reversed(Δ<0)",
                    "reversal_rate",
                    "avg_magnitude_reverse_pp(if_any)"])
        for key in sorted(by_cohort_wsi_rna.keys()):
            wc = by_cohort_wsi_rna[key].get("eval_with_censored", [])
            uo = by_cohort_wsi_rna[key].get("eval_uncensored_only", [])
            if not wc or not uo:
                continue
            mw, mu = st.mean(wc), st.mean(uo)
            delta = (mu - mw) * 100.0
            # 逐 run 配对（相同 fold/seed 最好；否则全排列一一对应近似）
            pairs = min(len(wc), len(uo))
            rev_cnt = 0
            mag = 0.0
            for a, b in zip(sorted(wc), sorted(uo)):
                d = (b - a) * 100.0
                if d < 0:
                    rev_cnt += 1
                    mag += abs(d)
            rate = (rev_cnt / pairs) if pairs else 0.0
            mag_avg = (mag / rev_cnt) if rev_cnt else 0.0
            cohort, wsi, rna = key
            w.writerow([
                cohort, wsi, rna,
                len(wc), f"{mw:.5f}", len(uo), f"{mu:.5f}",
                f"{delta:+.3f}", rev_cnt, f"{rate:.3f}", f"{mag_avg:.3f}",
            ])
    print(f"[DONE] {reversal_csv}（删失反转 Δ=uncens-with_cens，Δ<0 即「删失偏倚负向反转」）")

    # === Output 3: activation_window_quant.csv（用户 #6：激活窗口数据量化）===
    act_csv = SAVE_DIR / "activation_window_quant.csv"
    with act_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "wsi_model", "rna_mode", "window_L", "window_U", "A",
                    "N_runs_with_act",
                    "med_activation_epoch", "min_act_ep", "max_act_ep",
                    "med_C_pre", "med_C_post", "med_delta_post_pre_pp",
                    "count_C_post_ge_pre", "count_C_post_lt_pre", "win_rate"])
    buckets: dict[tuple[str, str, str, float, float, float], list[Row]] = {}
    for r in rows:
        if r.stage and r.stage != 2:
            continue
        if r.act_epoch is None:
            continue
        if r.window_l is None:
            continue
        k = (r.cohort or "UNK", r.wsi_model or "UNK", r.rna_mode or "UNK",
             r.window_l, r.window_u, r.window_A or 0.99)
        buckets.setdefault(k, []).append(r)
    with act_csv.open("a", newline="") as f:
        w = csv.writer(f)
        for k in sorted(buckets.keys()):
            rs = buckets[k]
            act_eps = [r.act_epoch for r in rs if r.act_epoch is not None]
            pres = [r.test_c_pre_act for r in rs if r.test_c_pre_act is not None]
            posts = [r.test_c_post_act for r in rs if r.test_c_post_act is not None]
            pairs = [(r.test_c_pre_act, r.test_c_post_act) for r in rs
                     if r.test_c_pre_act is not None and r.test_c_post_act is not None]
            win = sum(1 for (a, b) in pairs if b >= a)
            lose = sum(1 for (a, b) in pairs if b < a)
            delta_pairs = [(b - a) * 100.0 for (a, b) in pairs]
            cohort, wsi, rna, l, u, a = k
            w.writerow([
                cohort, wsi, rna, l, u, a,
                len(act_eps),
                int(st.median(act_eps)) if act_eps else -1,
                min(act_eps) if act_eps else -1,
                max(act_eps) if act_eps else -1,
                f"{st.median(pres):.4f}" if pres else "",
                f"{st.median(posts):.4f}" if posts else "",
                f"{st.median(delta_pairs):+.3f}" if delta_pairs else "",
                win, lose, f"{(win/(win+lose)):.3f}" if (win + lose) else "",
            ])
    print(f"[DONE] {act_csv}（C_post/C_pre 对比 + 激活 epoch 分布）")

    # === Output 4: aggregate.json ===
    agg = {
        "meta": {"generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                 "n_rows_total": len(rows)},
        "rows": [asdict(r) for r in rows],
        "ablation": list(csv.DictReader(open(f_csv))),
        "reversal": list(csv.DictReader(open(reversal_csv))),
        "activation": list(csv.DictReader(open(act_csv))),
    }
    jp = SAVE_DIR / "aggregate.json"
    json.dump(agg, open(jp, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"[DONE] {jp}（统一结构化文件，画图脚本直接 load）")

    # 打印摘要给 stdout 让用户 10 秒内扫完
    print("\n" + "=" * 120)
    print("📊 现有数据覆盖（已完成的 P2 runs）：")
    cohorts_done = sorted({r.cohort for r in rows if r.stage == 2 and r.cohort})
    for c in cohorts_done:
        ws = sorted({r.wsi_model for r in rows if r.cohort == c and r.stage == 2})
        print(f"   {c:<4s}  WSI = {ws}")
    print("-" * 120)
    print("📋 离店后 7 天：重跑本脚本 = 自动刷新全部输出（skip_existing 只扫新增完成 runs）")
    print(f"   运行命令：  cd /root/autodl-tmp/R2wsp && python scripts/analyze_ablation_aggregate.py")


if __name__ == "__main__":
    main()
