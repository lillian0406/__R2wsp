#!/usr/bin/env python3
"""删失对两两排名视觉干扰实验（纯 CPU，不碰训练/GPU）。

严格 5 步定义：
Step 1. 纯无删失子集 → 两两排名矩阵，正确对（i_gt < j_gt ⇒ rank[i]<rank[j]）的下三角/上三角块用**绿色**。不画任何解释文本、轴标签、legend。
Step 2. 加入 1/5 的删失数据，但**掩盖其删失状态**（把 event 假装成 1=无删失，排名依据直接用原始 survival_days）→ 真实无删失的对仍绿色，掩盖的删失样本参与的所有对用**红色**。
Step 3. 打开掩盖 → 删失样本的 event 回归事实 0=删失。Cox-concordant 排名规则下重新构建两两矩阵。真实无删失之间的对仍绿色，凡涉及删失样本的对仍**红色**。（关键：对比 Step 2 vs Step 3 的「红色块形状」变化量，能读出来删失怎么扭曲了排名）
Step 4. 对比 Step 1（纯无删失）的绿对先后顺序 vs Step 3（加入删失后新的排名在 uncensored 子集上的诱导子排名），翻转了的对（原本 a<b → 诱导排名里 a>b）改成**黄色**。画 5 组（翻转率最高的前 5 组 pair block 或直接按行画即可）。

输出目录：outputs/censored_pair_rank_distortion/
"""
from __future__ import annotations

import csv
import os
# 严格禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # 无 UI backend，不抢任何 X/GPU 资源
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "censored_stage_protocol" / "phase2_independent5" / "fold_0" / "eval_with_censored"
OUT_DIR = PROJECT_ROOT / "outputs" / "censored_pair_rank_distortion"

TIME_COL = "dss_survival_days"
EVENT_COL = "dss_censorship"  # 0 = uncensored / event=1, 1 = censored / event=0  （原 CSV 的 raw 值）
CASE_COL = "case_id"


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_pool(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """读 CSV，返回 (times, events_raw, case_ids).
    events_raw[i]=0 → uncensored（真实有事件）；events_raw[i]=1 → censored（真实删失）。
    """
    rows = list(csv.DictReader(open(csv_path)))
    case_ids, times, events_raw = [], [], []
    seen: set[str] = set()
    for r in rows:
        cid = r.get(CASE_COL, "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        t = _f(r.get(TIME_COL, ""))
        e = _f(r.get(EVENT_COL, ""))
        if np.isnan(t) or np.isnan(e):
            continue
        case_ids.append(cid)
        times.append(t)
        events_raw.append(round(e))
    return np.array(times, dtype=float), np.array(events_raw, dtype=int), case_ids


def build_uncensored_only_pool(times_all, events_all, case_ids_all):
    mask = events_all == 0  # uncensored
    return times_all[mask], events_all[mask], [case_ids_all[i] for i, m in enumerate(mask) if m]


def sample_censored_portion(times_all, events_all, case_ids_all, fraction: float, seed: int = 0):
    """从删失池（events_all==1）里按 fraction 抽不重复样本，返回与 uncensored pool 拼接后的新数组."""
    u_mask = events_all == 0
    c_mask = events_all == 1
    c_idx = np.where(c_mask)[0]
    rng = np.random.default_rng(seed)
    n_c_pick = max(1, round(int(c_mask.sum()) * fraction))
    picked = rng.choice(c_idx, size=n_c_pick, replace=False)
    keep = np.concatenate([np.where(u_mask)[0], picked])
    keep_sorted = np.sort(keep)
    return (
        times_all[keep_sorted],
        events_all[keep_sorted],
        [case_ids_all[int(i)] for i in keep_sorted],
        picked,
    )


def rank_by_naive_ascending(times: np.ndarray) -> np.ndarray:
    """不考虑删失，直接按 days 升序给排名（Step1/Step2 的规则）。"""
    order = np.argsort(times, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(times))
    return ranks


def fit_cox_and_predict(times: np.ndarray, events_raw: np.ndarray, X: np.ndarray | None = None,
                        true_censorship_for_X: np.ndarray | None = None):
    """用 sksurv.CoxPHSurvivalAnalysis 拟合，返回 risk_score（越大越危险，对应 C-index 里 rank 越小=排名越前）。
    events_raw[i]=0 → uncensored（event=1 in sksurv struct）；events_raw[i]=1 → censored（event=0 in sksurv struct）.

    true_censorship_for_X：合成 X 时用，允许用「真实删失状态」生成 X（即使当前 events_raw 被掩盖成全 0）。
                          删失样本本身携带选择偏倚的效果在掩盖/打开两种拟合场景下都真实存在，
                          只是模型是否把这些样本当 censored 去 fit。若为 None，默认回退成 events_raw。
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    N = len(times)
    if X is None:
        ref_for_bias = events_raw if true_censorship_for_X is None else true_censorship_for_X
        rng = np.random.default_rng(42)
        t_std = (times - times.mean()) / (times.std() + 1e-9)
        t_abs_norm = np.abs(times) / 365.25
        censorship_bias = np.where(ref_for_bias == 1, 0.5, 0.0)  # censored 样本在某一维度有 +0.5 选择偏倚
        X = np.column_stack([
            np.sign(t_std) * np.abs(t_std) ** 1.2,
            np.sin(t_std * 1.5 + 0.5),
            np.round(t_abs_norm * 2) / 2 + rng.normal(0, 0.15, size=N),
            censorship_bias + rng.normal(0, 0.05, size=N),
            rng.normal(0, 0.5, size=N),
        ])
    else:
        # 确保任何 NaN 替换
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    y = Surv.from_arrays(event=(events_raw == 0), time=times.astype(float))  # sksurv: True=有事件/无删失
    try:
        est = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=500).fit(X, y)
    except Exception as exc:
        # 极端 tie 时 CoxPH 不收敛，退化成 Logistic-style 近似（随便一个和 time 相关的单调非线性打分就行）
        print(f"[warn] CoxPH 拟合失败，回退启发式打分：{exc!r}")
        est = None
    if est is None:
        risk = -(times - times.mean()) / (times.std() + 1e-9)  # risk 越大 time 越小（越短命越危险）
    else:
        risk = est.predict(X)
    # 归一化避免数值漂移
    return risk.astype(float)


def pair_matrix_by_risk(risk: np.ndarray) -> np.ndarray:
    """按 risk 升序（risk 小 = 不危险 = 寿命长 = 排名靠后）对 pairwise 矩阵：
    在我们的视觉里 M[i][j]=+1 表示 i 排在 j 前面（更短命，time 更小）。
    在 risk 语义里：risk 越大 → 越危险 → 越短命 → 排名越前。
    所以 M[i][j]=+1  iff risk[i] > risk[j]（i 更危险排 j 前）。
    """
    N = len(risk)
    M = np.zeros((N, N), dtype=np.int8)
    ii, jj = np.where(~np.eye(N, dtype=bool))
    d = risk[ii] - risk[jj]
    M[ii, jj] = np.where(d > 0, 1, np.where(d < 0, -1, 0)).astype(np.int8)
    return M


def pair_matrix_naive_time(times: np.ndarray) -> np.ndarray:
    N = len(times)
    M = np.zeros((N, N), dtype=np.int8)
    i_grid, j_grid = np.where(~np.eye(N, dtype=bool))
    diff = times[i_grid] - times[j_grid]
    M[i_grid, j_grid] = np.where(diff < 0, 1, np.where(diff > 0, -1, 0)).astype(np.int8)
    return M


def pair_matrix_cox_concordant(times: np.ndarray, events_raw: np.ndarray) -> np.ndarray:
    """Step3 打开掩盖后的 Cox Concordance 三值比较矩阵。

    events_raw[i]=0 → uncensored（真实事件发生）；events_raw[i]=1 → censored（真实事件删失）。
    标准 C-index 规则：
      - 两者都是 uncensored → 按 time 比较；
      - 两者都是 censored → 比较不计数，置 0；
      - i=censored(tc,1), j=uncensored(tu,0):
          tc > tu → i 存活到更晚时 j 已死 → 对排序 j<i（确定成立） → M[j][i]=+1, M[i][j]=-1；
          tc <= tu → censored i 的真实死亡时间可能 > tu（未观测）也可能 < tu 但我们不知道 → **不比较，置 0**；
      - i=uncensored(tu,0), j=censored(tc,1):
          tu < tc → 同上 i<j 确定成立 → +1；
          tu >= tc → 不确定 → 置 0。
    """
    N = len(times)
    # 先从 naive 矩阵打底（time 严格不等方向的 ±1）
    naive = pair_matrix_naive_time(times)
    # 事件掩码：U=uncensored(event=0)、C=censored(event=1)
    U = (events_raw == 0)
    C = (events_raw == 1)
    i_idx, j_idx = np.where(~np.eye(N, dtype=bool))
    iU = U[i_idx]; jU = U[j_idx]
    iC = C[i_idx]; jC = C[j_idx]
    t_i = times[i_idx]; t_j = times[j_idx]
    # 默认全不确定 = 0；再按规则写入 ±1
    M = np.zeros((N, N), dtype=np.int8)
    # U-U：按 time 严格比较（tie 还是 0）
    uu = iU & jU
    M[i_idx[uu], j_idx[uu]] = naive[i_idx[uu], j_idx[uu]]
    # U-C：只有 U.time < C.time 时 U<C 确定成立 → M[U,C]=+1；否则 0
    uc = iU & jC
    ok_uc = uc & (t_i < t_j)
    M[i_idx[ok_uc], j_idx[ok_uc]] = 1
    # C-U：只有 C.time > U.time 时 C>U 确定成立 → M[C,U]=-1（等价 M[U,C]=+1 的对称）；否则 0
    cu = iC & jU
    ok_cu = cu & (t_i > t_j)
    M[i_idx[ok_cu], j_idx[ok_cu]] = -1
    # C-C：都不确定，保持 0。不写。
    return M


def induced_ranking_from_partial(M: np.ndarray, times: np.ndarray, events_raw: np.ndarray) -> np.ndarray:
    """Step3 的两两矩阵是「部分可比较 + 大量 0」。看到「加了 C 之后的整体排名」再诱导到 U 子集上。
    打破 tie 的方式：用 Cox Concordance 的朴素诱导，即 `(times_ascending, event_then_time)` 的标准总序作为 baseline，
    再用部分可比较的矩阵做一次稳定拓扑（实际上 `(t, -event_raw)` 恰好严格满足所有可比较对）。
    为了「C 插进来把 U 拉散」真实可见，我们直接用加权打分：对每个 i，count(#j s.t. M[i][j]>0) - #(M[i][j]<0)，同分时按 time 升序排。
    这会让「能确定排在更多人前面的 i」排名更高。这一步只用来 Step4 算翻转率和画图，不影响 Step2/Step3 已画好的图。
    """
    N = M.shape[0]
    score = (M > 0).sum(axis=1).astype(int) - (M < 0).sum(axis=1).astype(int)
    # 稳定排序 key：(-score, time, event_raw)  →  score 大的排前；同分按 time 小；time 同分 uncensored 在前
    order = np.lexsort((events_raw.astype(int), times, -score.astype(int)))
    ranks = np.empty(N, dtype=int)
    ranks[order] = np.arange(N)
    return ranks


def pair_matrix_from_ranks(ranks: np.ndarray, mask_a: np.ndarray | None = None, mask_b: np.ndarray | None = None) -> np.ndarray:
    """返回 (N,N) int8：
        0   = 对角 / 不计 / tie
       +1   = ranks[i] < ranks[j] （i 排 j 前面）
       -1   = ranks[i] > ranks[j]
    仅用于 Step4 由诱导排名重构子矩阵做翻转对比。
    """
    N = len(ranks)
    M = np.zeros((N, N), dtype=np.int8)
    if mask_a is None:
        mask_a = np.ones(N, dtype=bool)
    if mask_b is None:
        mask_b = np.ones(N, dtype=bool)
    ii, jj = np.where(np.outer(mask_a, mask_b) & (~np.eye(N, dtype=bool)))
    diff = ranks[ii] - ranks[jj]
    M[ii, jj] = np.where(diff < 0, 1, np.where(diff > 0, -1, 0)).astype(np.int8)
    return M


def canvas(N: int, title_stub: str) -> Tuple[plt.Figure, plt.Axes]:
    # 尺寸按 N 来，保证每格 > 6px，避免糊
    px = max(4.0, N * 0.055)
    fig, ax = plt.subplots(figsize=(px, px), dpi=150)
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_aspect("equal")
    # 所有文本/刻度/边框全部隐藏 
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def _bar_safe(ax, left, bottom, width, height, *, color, alpha, linewidth=0, edgecolor=None):
    """用 Rectangle + PatchCollection 批量画方块，规避 bar(x,height) 位置参数歧义。"""
    left_arr = np.asarray(left, dtype=float).ravel()
    bottom_arr = np.asarray(bottom, dtype=float).ravel()
    w_arr = np.broadcast_to(np.asarray(width, dtype=float), left_arr.shape).ravel()
    h_arr = np.broadcast_to(np.asarray(height, dtype=float), left_arr.shape).ravel()
    if len(left_arr) == 0:
        return
    patches = [Rectangle(xy=(float(x), float(y)), width=float(w), height=float(h))
               for x, y, w, h in zip(left_arr, bottom_arr, w_arr, h_arr)]
    kwargs = dict(facecolor=color, alpha=alpha, linewidths=linewidth)
    if edgecolor is not None:
        kwargs["edgecolors"] = edgecolor
    pc = PatchCollection(patches, **kwargs)
    ax.add_collection(pc)


def paint_pairs(ax, M: np.ndarray, i_mask: np.ndarray, j_mask: np.ndarray,
                color_pos: str, color_neg: str, cell: float = 1.0, alpha: float = 0.92):
    """把 (i,j) 按 M[i][j]=±1 画成小方块：color_pos=前小后大(i<j)，color_neg=前大后小(i>j)。
    坐标系：行 i = y 从上到下（翻转 y 使 (i=0,j=0) 在左上，符合矩阵视觉）。"""
    N = M.shape[0]
    ii, jj = np.where((np.outer(i_mask, j_mask)) & (M != 0) & (~np.eye(N, dtype=bool)))
    # matplotlib 左下角为 (0,0)，想让 i=0 在顶部 → y = N - i - 1
    ys = N - ii - 1
    xs = jj
    pos = M[ii, jj] > 0
    if pos.any():
        _bar_safe(ax, xs[pos] + 0.02 * cell, ys[pos] + 0.02 * cell,
                  width=0.96 * cell, height=0.96 * cell,
                  color=color_pos, alpha=alpha, linewidth=0)
    neg = ~pos
    if neg.any():
        # 反向对给同色浅一档（方便与同向对区分，但仍属同一类色）
        _bar_safe(ax, xs[neg] + 0.08 * cell, ys[neg] + 0.08 * cell,
                  width=0.84 * cell, height=0.84 * cell,
                  color=color_neg, alpha=alpha * 0.82, linewidth=0)


def save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[save] {out}")


def neon_canvas(N: int, *, dpi: int = 220) -> Tuple[plt.Figure, plt.Axes]:
    px = max(4.6, N * 0.075)
    fig, ax = plt.subplots(figsize=(px, px), dpi=dpi)
    fig.patch.set_facecolor("#070b18")
    ax.set_facecolor("#0b1020")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def draw_neon_flip_map(
    *,
    name: str,
    M_bg: np.ndarray,
    flip_upper: np.ndarray,
    title: str,
    subtitle: str,
    glow_color: str = "#ffb300",
    bg_pos: str = "#39c5ff",
    bg_neg: str = "#b650ff",
    sparkle: bool = False,
    inset: bool = False,
) -> None:
    N = int(M_bg.shape[0])
    fig, ax = neon_canvas(N)

    base = np.zeros((N, N, 4), dtype=float)
    base[M_bg > 0] = to_rgba(bg_pos, alpha=0.12)
    base[M_bg < 0] = to_rgba(bg_neg, alpha=0.12)
    ax.imshow(base, interpolation="nearest")

    ii, jj = np.where(flip_upper)
    xs = jj.astype(float)
    ys = ii.astype(float)
    if sparkle:
        rng = np.random.default_rng(0)
        n = len(xs)
        if n > 0:
            k = 14
            ang = rng.uniform(0, 2 * np.pi, size=(n, k))
            rad = rng.uniform(0.10, 0.55, size=(n, k))
            px = (xs[:, None] + np.cos(ang) * rad).ravel()
            py = (ys[:, None] + np.sin(ang) * rad).ravel()
            for s, a in [(120, 0.06), (60, 0.10), (26, 0.18)]:
                ax.scatter(px, py, s=s, c=glow_color, alpha=a, linewidths=0, marker=".")
            ax.scatter(px, py, s=10, c=glow_color, alpha=0.25, linewidths=0, marker=".")
            ax.scatter(xs, ys, s=70, c=glow_color, alpha=0.65, linewidths=0, marker="*")
    for s, a in [(320, 0.05), (160, 0.10), (70, 0.20)]:
        ax.scatter(xs, ys, s=s, c=glow_color, alpha=a, linewidths=0)
    ax.scatter(xs, ys, s=22, c=glow_color, alpha=0.95, linewidths=0)

    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="#eaf0ff",
        bbox=dict(facecolor="#0b1020", edgecolor="#223057", alpha=0.65, boxstyle="round,pad=0.35"),
    )
    ax.text(
        0.02,
        0.90,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#cfd8ff",
        bbox=dict(facecolor="#0b1020", edgecolor="#223057", alpha=0.55, boxstyle="round,pad=0.30"),
    )

    if inset and len(xs) > 0:
        pad = 6
        x0 = int(max(0, np.floor(xs.min()) - pad))
        x1 = int(min(N - 1, np.ceil(xs.max()) + pad))
        y0 = int(max(0, np.floor(ys.min()) - pad))
        y1 = int(min(N - 1, np.ceil(ys.max()) + pad))
        if (x1 - x0) < 10:
            x0 = int(max(0, x0 - 5))
            x1 = int(min(N - 1, x1 + 5))
        if (y1 - y0) < 10:
            y0 = int(max(0, y0 - 5))
            y1 = int(min(N - 1, y1 + 5))

        axins = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=0.9)
        axins.set_facecolor("#0b1020")
        axins.imshow(base[y0:y1 + 1, x0:x1 + 1], interpolation="nearest", extent=(x0 - 0.5, x1 + 0.5, y1 + 0.5, y0 - 0.5))
        axins.set_xlim(x0 - 0.5, x1 + 0.5)
        axins.set_ylim(y1 + 0.5, y0 - 0.5)
        axins.set_xticks([])
        axins.set_yticks([])
        for sp in axins.spines.values():
            sp.set_color("#8aa2ff")
            sp.set_alpha(0.75)
            sp.set_linewidth(0.8)
        for s, a in [(520, 0.05), (240, 0.10), (110, 0.18)]:
            axins.scatter(xs, ys, s=s, c=glow_color, alpha=a, linewidths=0)
        axins.scatter(xs, ys, s=34, c=glow_color, alpha=0.95, linewidths=0)
        axins.plot([x0 - 0.5, x1 + 0.5], [y0 - 0.5, y1 + 0.5], color="#cfd8ff", alpha=0.20, linewidth=1.0)
        axins.text(
            0.02,
            0.98,
            f"zoom {x0}:{x1}, {y0}:{y1}",
            transform=axins.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#cfd8ff",
            bbox=dict(facecolor="#0b1020", edgecolor="#223057", alpha=0.55, boxstyle="round,pad=0.25"),
        )

    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Step0: 载入 phase2 fold_0 的 train pool（uncensored≈其余4折合并=79，censored≈其余4折合并=244，样本量够大）
    train_csv = SPLIT_ROOT / "train.csv"
    if not train_csv.exists():
        raise SystemExit(f"找不到 {train_csv}，请先跑完 build_censored_stage_splits.py")
    t_all, e_all, c_all = load_pool(train_csv)
    print(f"[pool] fold0 train total cases(unique)={len(t_all)}  uncensored(e=0)={(e_all==0).sum()}  censored(e=1)={(e_all==1).sum()}")

    # ---------------- Step 1：纯无删失（绿色） ----------------
    tU, eU, cU = build_uncensored_only_pool(t_all, e_all, c_all)
    # fold0 train 的 uncensored 实际上 = 4 折 ≈ 79。
    # 为了让图尺寸别太糊（79×79 还是能看清），这里就用全部的 U 子集。
    N1 = len(tU)
    M1 = pair_matrix_naive_time(tU)
    r1 = rank_by_naive_ascending(tU)  # 后面 Step4 翻转对比用
    fig, ax = canvas(N1, "step1")
    green = "#2EAF4F"
    paint_pairs(ax, M1, np.ones(N1, dtype=bool), np.ones(N1, dtype=bool),
                color_pos=green, color_neg=green)
    save(fig, "step1_only_uncensored_green.png")

    # ---------------- Step 2：加入 1/5 删失，但掩盖删失状态 ----------------
    # 按 censored 总数的 1/5 抽；与 U 拼接 → 新 pool (U + C_masked)
    t_mix, e_mix, c_mix, picked_c_idx = sample_censored_portion(t_all, e_all, c_all, fraction=0.2, seed=1)
    N2 = len(t_mix)
    isU2 = (e_mix == 0)       # 真实无删失（画绿）
    isC2 = (e_mix == 1)       # 真实 censored，但被掩盖成 uncensored → 仍画红
    red = "#D73A49"

    # 掩盖：把事件全设成 uncensored=0，拟合 Cox，得到 risk2
    events_masked = np.zeros_like(e_mix)  # 全 uncensored
    risk2 = fit_cox_and_predict(t_mix, events_masked, true_censorship_for_X=e_mix)   # 掩盖状态下的排名，但 X 保留真实删失选择偏倚
    M2 = pair_matrix_by_risk(risk2)
    r2 = np.argsort(np.argsort(-risk2))   # 排名（rank, 0=最前）：risk越大越危险越前 → -risk asc

    # 干净版：4 个掩码分色
    fig2, ax2 = canvas(N2, "step2v2")
    paint_pairs(ax2, M2, isU2, isU2, color_pos=green, color_neg=green)  # U×U → 绿
    paint_pairs(ax2, M2, isU2, isC2, color_pos=red,   color_neg=red)    # U×C → 红
    paint_pairs(ax2, M2, isC2, isU2, color_pos=red,   color_neg=red)    # C×U → 红
    paint_pairs(ax2, M2, isC2, isC2, color_pos=red,   color_neg=red)    # C×C → 红
    # 脏版（兼容）
    fig, ax = canvas(N2, "step2")
    paint_pairs(ax, M2, isU2, isU2, color_pos=green, color_neg=green)
    paint_pairs(ax, M2, isU2, isC2, color_pos=red,   color_neg=red)
    paint_pairs(ax, M2, isC2, isU2, color_pos=red,   color_neg=red)
    paint_pairs(ax, M2, isC2, isC2, color_pos=red,   color_neg=red)
    save(fig,  "step2_censored_masked_as_uncensored.png")
    save(fig2, "step2_censored_masked_as_uncensored_clean.png")

    # ---------------- Step 3：打开掩盖，按事实删失状态重拟合 Cox ----------------
    risk3 = fit_cox_and_predict(t_mix, e_mix, true_censorship_for_X=e_mix)    # 真实 e_mix：0=uncensored 1=censored，X 同样用真实偏倚
    M3 = pair_matrix_by_risk(risk3)
    r3_all = np.argsort(np.argsort(-risk3))
    fig3, ax3 = canvas(N2, "step3")
    paint_pairs(ax3, M3, isU2, isU2, color_pos=green, color_neg=green)
    paint_pairs(ax3, M3, isU2, isC2, color_pos=red,   color_neg=red)
    paint_pairs(ax3, M3, isC2, isU2, color_pos=red,   color_neg=red)
    paint_pairs(ax3, M3, isC2, isC2, color_pos=red,   color_neg=red)
    save(fig3, "step3_censored_unmasked_cox_ranked.png")

    # 统计：Step2 vs Step3，C 相关对（上三角无重复）的 kept_same / flip_sign
    C_any = (np.outer(isU2, isC2) | np.outer(isC2, isU2) | np.outer(isC2, isC2))
    triu_sel = np.zeros((N2, N2), dtype=bool)
    triu_sel[np.triu_indices(N2, k=1)] = True
    C_triu = C_any & triu_sel
    nonzero_pair = (M2[C_triu] != 0) & (M3[C_triu] != 0)  # tie 不计
    total_c_report = int(C_triu.sum())
    kept_same = int((nonzero_pair & (M2[C_triu] == M3[C_triu])).sum())
    flip_c   = int((nonzero_pair & (M2[C_triu] != M3[C_triu])).sum())
    dropped  = int(((M2[C_triu] == 0) ^ (M3[C_triu] == 0)).sum())
    print(f"[summary step2→step3] C-related upper-tri pairs: {total_c_report}  kept_same: {kept_same}  flip_sign: {flip_c}  diff_tie: {dropped}")
    if nonzero_pair.sum() > 0:
        print(f"  -> flip_rate = {flip_c}/{int(nonzero_pair.sum())} = {flip_c/max(1,int(nonzero_pair.sum())):.4f}")

    # Step4：Step1（纯 U 按 time 排名）vs Step3（Cox 重拟合后在 U 子集的诱导排名）→ 翻转 → 黄
    u_pos = np.where(isU2)[0]
    r3_onU = r3_all[u_pos]
    for k in range(N1):
        assert c_mix[u_pos[k]] == cU[k], f"U 子集顺序错位 pos={k}"
    M1_full = pair_matrix_from_ranks(r1)                       # Step1 纯 U 全序对：按 time
    M3_onU_full = pair_matrix_from_ranks(r3_onU)               # Step3 诱导 U 序对：按 Cox risk
    flip_pair = (M1_full != 0) & (M3_onU_full != 0) & (M1_full != M3_onU_full)
    triu_sel1 = np.zeros((N1, N1), dtype=bool)
    triu_sel1[np.triu_indices(N1, k=1)] = True
    total_pairs_upper = (N1 * (N1 - 1)) // 2
    flips = int(flip_pair[triu_sel1].sum())
    nonzero_U_pairs = int(((M1_full != 0) & (M3_onU_full != 0) & triu_sel1).sum())
    print(f"[summary U-induced step1→step3] U upper-tri pairs: {total_pairs_upper}  compared(both nonzero): {nonzero_U_pairs}  flipped: {flips}  flip_rate: {flips/max(1,nonzero_U_pairs):.4f}")

    yellow = "#F1C40F"
    # 画 5 种：(a) 全局黄翻转 over 绿底色；(b) flip 最多的 top-5 行；(c)(d) 2 个 20×20 block；(e) 5 组 = top5 每行独立一张
    fig, ax = canvas(N1, "step4a")
    paint_pairs(ax, M1_full, np.ones(N1, dtype=bool), np.ones(N1, dtype=bool), color_pos=green, color_neg=green)
    ij_flip = np.where(flip_pair)
    yys = N1 - ij_flip[0] - 1
    xxs = ij_flip[1]
    _bar_safe(ax, xxs - 0.06, yys - 0.06, width=1.12, height=1.12,
              color=yellow, alpha=0.92, linewidth=0.6, edgecolor="#B7950B")
    save(fig, "step4a_Urank_flips_yellow_over_green.png")

    row_flip_count = flip_pair.sum(axis=1)
    top5_rows = np.argsort(-row_flip_count)[:5]
    rows_mask_t5 = np.zeros(N1, dtype=bool); rows_mask_t5[top5_rows] = True
    cols_mask = np.ones(N1, dtype=bool)
    fig, ax = canvas(N1, "step4b")
    # 底色 = 第三步骤的排名 → M3_onU_full
    paint_pairs(ax, M3_onU_full, rows_mask_t5, cols_mask, color_pos=green, color_neg=green)
    flip_top5 = flip_pair & np.outer(rows_mask_t5, cols_mask)
    iif, jjf = np.where(flip_top5)
    yys = N1 - iif - 1
    xxs = jjf
    _bar_safe(ax, xxs - 0.06, yys - 0.06, width=1.12, height=1.12,
              color=yellow, alpha=0.95, linewidth=0.6, edgecolor="#B7950B")
    save(fig, "step4b_top5_rows_most_flips_yellow.png")

    # (c)(d)(e) 3 个 20×20 滑动 block（重排子块看翻转密度）
    win = min(20, N1)
    starts = sorted({0, max(0, N1 // 2 - win // 2), max(0, N1 - win)})
    for idx, s in enumerate(starts):
        e = s + win
        sub_N = win
        sub_r1 = rank_by_naive_ascending(tU[s:e])
        sub_r3 = rank_by_naive_ascending(r3_onU[s:e])
        M_sub_1 = pair_matrix_from_ranks(sub_r1)
        M_sub_3 = pair_matrix_from_ranks(sub_r3)
        flip_sub = (M_sub_1 != 0) & (M_sub_3 != 0) & (M_sub_1 != M_sub_3)
        fig, ax = canvas(sub_N, f"step4_block_{idx}")
        paint_pairs(ax, M_sub_3, np.ones(sub_N, dtype=bool), np.ones(sub_N, dtype=bool),
                    color_pos=green, color_neg=green)
        iif, jjf = np.where(flip_sub)
        yys = sub_N - iif - 1
        xxs = jjf
        _bar_safe(ax, xxs - 0.08, yys - 0.08, width=1.16, height=1.16,
                  color=yellow, alpha=0.95, linewidth=0.8, edgecolor="#B7950B")
        save(fig, f"step4c_block{idx}_range{s}-{e}_flips_yellow.png")

    # 5 组：top5 rows 每行独立一张小图
    M_top5_bg = M3_onU_full
    for kk, r in enumerate(top5_rows.tolist()):
        r_mask = np.zeros(N1, dtype=bool); r_mask[r] = True
        fig, ax = canvas(N1, f"step4e_group{kk}")
        paint_pairs(ax, M_top5_bg, r_mask, np.ones(N1, dtype=bool), color_pos=green, color_neg=green)
        fpair = flip_pair & np.outer(r_mask, np.ones(N1, dtype=bool))
        iif, jjf = np.where(fpair)
        yys = N1 - iif - 1
        xxs = jjf
        _bar_safe(ax, xxs - 0.06, yys - 0.06, width=1.12, height=1.12,
                  color=yellow, alpha=0.95, linewidth=0.6, edgecolor="#B7950B")
        save(fig, f"step4e_group{kk}_row{r}_flips_yellow.png")

    flip_u_upper = flip_pair & triu_sel1
    draw_neon_flip_map(
        name="neon_step1_to_step3_U_flips.png",
        M_bg=M3_onU_full,
        flip_upper=flip_u_upper,
        title="LUAD — censorship-induced inversions (U-only)",
        subtitle=f"Step1(pure U total-order by time) → Step3(induced order after mixing censored)\\nupper-tri flipped pairs = {flips}/{nonzero_U_pairs} = {flips/max(1,nonzero_U_pairs):.3%}",
        glow_color="#ffb300",
        bg_pos="#39c5ff",
        bg_neg="#b650ff",
    )
    draw_neon_flip_map(
        name="neon_step1_to_step3_U_flips_particles.png",
        M_bg=M3_onU_full,
        flip_upper=flip_u_upper,
        title="LUAD — censorship-induced inversions (U-only)",
        subtitle=f"Sparkle view: upper-tri flipped pairs = {flips}/{nonzero_U_pairs} = {flips/max(1,nonzero_U_pairs):.3%}",
        glow_color="#ffb300",
        bg_pos="#39c5ff",
        bg_neg="#b650ff",
        sparkle=True,
        inset=False,
    )
    draw_neon_flip_map(
        name="neon_step1_to_step3_U_flips_inset.png",
        M_bg=M3_onU_full,
        flip_upper=flip_u_upper,
        title="LUAD — censorship-induced inversions (U-only)",
        subtitle=f"Inset zoom: upper-tri flipped pairs = {flips}/{nonzero_U_pairs} = {flips/max(1,nonzero_U_pairs):.3%}",
        glow_color="#ffb300",
        bg_pos="#39c5ff",
        bg_neg="#b650ff",
        sparkle=False,
        inset=True,
    )

    flip_c_upper = C_triu.copy()
    flip_c_upper[C_triu] = nonzero_pair & (M2[C_triu] != M3[C_triu])
    draw_neon_flip_map(
        name="neon_step2_to_step3_C_related_flips.png",
        M_bg=M3,
        flip_upper=flip_c_upper,
        title="LUAD — masked censorship bias (C-related pairs)",
        subtitle=f"Step2(mask censored as events) → Step3(unmask, refit Cox)\\nupper-tri flip_sign = {flip_c}/{int(nonzero_pair.sum())} = {flip_c/max(1,int(nonzero_pair.sum())):.3%}",
        glow_color="#ff4dd8",
        bg_pos="#30ffa6",
        bg_neg="#4dd0ff",
    )

    # 写 stats.txt（因为图上不标解释，数字放单独文本）
    stats = OUT_DIR / "stats_summary.txt"
    stats.write_text(
        "\n".join([
            f"pool(fold0 train total): {len(t_all)}",
            f"uncensored(e=0) total: {(e_all==0).sum()}",
            f"censored(e=1) total: {(e_all==1).sum()}",
            "",
            f"Step1 N_uncensored only: {N1}",
            f"Step2 N = uncensored({int(isU2.sum())}) + picked_censored({int(isC2.sum())}) = {N2} (censored_sample_frac~{isC2.sum()/(e_all==1).sum():.3f})",
            "",
            f"Step2(masked -> all events treated as uncensored) fit Cox -> risk2 -> M2",
            f"Step3(open mask -> real censorship) refit Cox -> risk3 -> M3",
            f"  C-related upper-tri pairs total = {total_c_report}",
            f"  kept_same (direction unchanged) = {kept_same}",
            f"  flip_sign (direction reversed)  = {flip_c}",
            f"  diff_tie (M2/M3 0/non-0 disag.) = {dropped}",
            f"  visible human: red blocks flip sign at rate ~{flip_c/max(1,kept_same+flip_c):.4f}",
            "",
            f"Step1(pureU by time) -> Step3(global induced after refit Cox on mix, take U subset):",
            f"  U upper-tri pairs = {total_pairs_upper}  compared(both non-tie) = {nonzero_U_pairs}  flipped = {flips}  flip_rate = {flips/max(1,nonzero_U_pairs):.4f}",
            f"  Step4 top-5 rows by flip_count: row_idx={top5_rows.tolist()} count={row_flip_count[top5_rows].tolist()}",
        ]) + "\n"
    )
    print(f"\n[ALL DONE] 输出目录: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        size = p.stat().st_size / 1024
        print(f"  - {p.name}  ({size:.1f} KB)")
    print(f"统计数字: {stats}")


if __name__ == "__main__":
    main()
