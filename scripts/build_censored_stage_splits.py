from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@dataclass(frozen=True)
class SplitBundle:
    root: Path
    train_csv: Path
    test_csv: Path
    manifest: dict


def read_split(root: Path):
    with (root / "train.csv").open() as f:
        train = list(csv.DictReader(f))
    with (root / "test.csv").open() as f:
        test = list(csv.DictReader(f))
    return train, test


def _censorship_col_for(target_col: str) -> str:
    prefix = target_col.split("_")[0]
    return f"{prefix}_censorship"


def _time_col_for(target_col: str) -> str:
    return target_col


def filter_uncensored(rows: list[dict], target_col: str) -> list[dict]:
    censor_col = _censorship_col_for(target_col)
    return [r for r in rows if float(r[censor_col]) < 0.5]


def filter_censored(rows: list[dict], target_col: str) -> list[dict]:
    censor_col = _censorship_col_for(target_col)
    return [r for r in rows if float(r[censor_col]) >= 0.5]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_kfold_case_ids(
    rows: list[dict],
    target_col: str,
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    """Stratified by censorship + time_bin on case_ids."""
    cases: dict[str, dict] = {}
    for r in rows:
        cid = str(r["case_id"])
        if cid not in cases:
            cases[cid] = r
    case_rows = list(cases.values())
    times = np.asarray([float(r[_time_col_for(target_col)]) for r in case_rows])
    cens = np.asarray([float(r[_censorship_col_for(target_col)]) for r in case_rows])
    unc = times[cens < 0.5]
    if unc.size >= n_folds * 2:
        bins = np.quantile(unc, np.linspace(0.0, 1.0, n_folds + 1))
        bins = np.unique(bins)
    else:
        bins = np.asarray([np.inf])
    if bins.size >= 2:
        time_bin = np.digitize(times, bins[1:-1], right=False)
    else:
        time_bin = np.zeros_like(times, dtype=np.int64)
    strata = [f"{int(c)}{int(tb)}" for c, tb in zip(cens.tolist(), time_bin.tolist())]
    fold_ids = [[] for _ in range(n_folds)]
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(strata):
        groups[s].append(idx)
    for key in sorted(groups.keys()):
        idx = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(idx)
        for j, i in enumerate(idx.tolist()):
            fold_ids[j % n_folds].append(str(case_rows[i]["case_id"]))
    return fold_ids


def rows_from_case_ids(all_rows: list[dict], case_ids: set[str]) -> list[dict]:
    return [r for r in all_rows if str(r["case_id"]) in case_ids]


def deduplicate_cases(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        cid = str(r["case_id"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


def union_official_train_test(official_splits_root: Path, target_col: str):
    """把官折 k=0..4 的 train/test 合并成全局 case 池，Phase-2 的独立 5 折从这里抽（和官折不一样，是新的“重新5折划分”）"""
    fold_roots = sorted(
        p for p in official_splits_root.glob("k=*") if p.is_dir()
    )
    all_rows: list[dict] = []
    for fr in fold_roots:
        train, test = read_split(fr)
        all_rows.extend(train)
        all_rows.extend(test)
    # deduplicate by case_id
    by_case: dict[str, dict] = {}
    for r in all_rows:
        cid = str(r["case_id"])
        if cid in by_case:
            continue
        by_case[cid] = r
    case_rows = list(by_case.values())
    uncensored = filter_uncensored(case_rows, target_col)
    censored = filter_censored(case_rows, target_col)
    return case_rows, uncensored, censored


def build_phase1_outer_splits(
    official_splits_root: Path,
    out_root: Path,
    target_col: str,
    n_outer: int,
    seed: int,
) -> list[SplitBundle]:
    bundles: list[SplitBundle] = []
    for k in range(n_outer):
        fold_root = official_splits_root / f"k={k}"
        if not fold_root.exists():
            raise FileNotFoundError(fold_root)
        train_all, test_all = read_split(fold_root)
        train_u = filter_uncensored(train_all, target_col)
        test_u = filter_uncensored(test_all, target_col)
        out_dir = out_root / "phase1_outer5" / f"fold_{k}"
        write_csv(out_dir / "train.csv", train_u)
        write_csv(out_dir / "test.csv", test_u)
        manifest = {
            "official_fold": k,
            "target_col": target_col,
            "filter": "uncensored_only_train_val_test",
            "n_train_cases_all": len(set(r["case_id"] for r in train_all)),
            "n_test_cases_all": len(set(r["case_id"] for r in test_all)),
            "n_train_cases_uncensored": len(set(r["case_id"] for r in train_u)),
            "n_test_cases_uncensored": len(set(r["case_id"] for r in test_u)),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        bundles.append(SplitBundle(out_dir, out_dir / "train.csv", out_dir / "test.csv", manifest))
    return bundles


def build_phase2_independent_5fold_splits(
    official_splits_root: Path,
    out_root: Path,
    target_col: str,
    n_folds: int,
    seed: int,
) -> None:
    """Phase-2：对全局 pool 重新做“uncensored 独立 5 折 + censored 对应 5 份不重复均分”。每折生成两套 eval 口径：
    - eval_with_censored：折内 test.csv 同时包含 uncensored 和 censored（和训练任务口径一致，默认用这套）
    - eval_uncensored_only：折内 test.csv 只保留 uncensored（做你说的“都试试看吧”的对照组）
    """
    _all_rows, uncensored_rows, censored_rows = union_official_train_test(
        official_splits_root, target_col
    )
    u_folds = stratified_kfold_case_ids(
        deduplicate_cases(uncensored_rows), target_col=target_col, n_folds=n_folds, seed=seed
    )
    c_folds = stratified_kfold_case_ids(
        deduplicate_cases(censored_rows), target_col=target_col, n_folds=n_folds, seed=seed + 9999
    )
    all_cases_rows = {str(r["case_id"]): r for r in _all_rows}

    top_root = out_root / "phase2_independent5"
    top_root.mkdir(parents=True, exist_ok=True)
    (top_root / "manifest.json").write_text(
        json.dumps(
            {
                "target_col": target_col,
                "n_folds": int(n_folds),
                "split_kind": "independent5_not_inherited_from_official",
                "uncensored_total_cases": len({r["case_id"] for r in uncensored_rows}),
                "censored_total_cases": len({r["case_id"] for r in censored_rows}),
                "uncensored_folds": [sorted(f) for f in u_folds],
                "censored_folds": [sorted(f) for f in c_folds],
                "per_fold_case_counts": [
                    {
                        "fold": int(i),
                        "uncensored": len(u_folds[i]),
                        "censored": len(c_folds[i]),
                    }
                    for i in range(n_folds)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    for i in range(n_folds):
        train_u_cases = set()
        test_u_cases = set()
        train_c_cases = set()
        test_c_cases = set()
        for j in range(n_folds):
            if j == i:
                test_u_cases.update(u_folds[j])
                test_c_cases.update(c_folds[j])
            else:
                train_u_cases.update(u_folds[j])
                train_c_cases.update(c_folds[j])
        train_joint_cases = train_u_cases | train_c_cases
        # --- Design A: eval WITH censorship（默认，和训练口径一致）---
        test_with_censor_cases = test_u_cases | test_c_cases
        train_rows = [all_cases_rows[cid] for cid in sorted(train_joint_cases)]
        test_with_cens_rows = [all_cases_rows[cid] for cid in sorted(test_with_censor_cases)]
        rootA = top_root / f"fold_{i}" / "eval_with_censored"
        write_csv(rootA / "train.csv", train_rows)
        write_csv(rootA / "test.csv", test_with_cens_rows)
        (rootA / "manifest.json").write_text(
            json.dumps(
                {
                    "fold": int(i),
                    "eval_kind": "eval_with_censored",
                    "train_cases_total": len(train_joint_cases),
                    "train_uncensored_cases_n": len(train_u_cases),
                    "train_censored_cases_n": len(train_c_cases),
                    "test_cases_total": len(test_with_censor_cases),
                    "test_uncensored_cases_n": len(test_u_cases),
                    "test_censored_cases_n": len(test_c_cases),
                    "notes": "用户新说的“折内 val/test 也加删失，和训练任务口径一致”的默认设计",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        # --- Design B: eval uncensored only（对照组）---
        test_uncens_only_rows = [all_cases_rows[cid] for cid in sorted(test_u_cases)]
        rootB = top_root / f"fold_{i}" / "eval_uncensored_only"
        write_csv(rootB / "train.csv", train_rows)  # train 完全一样，只是 test 的 filter 不同
        write_csv(rootB / "test.csv", test_uncens_only_rows)
        (rootB / "manifest.json").write_text(
            json.dumps(
                {
                    "fold": int(i),
                    "eval_kind": "eval_uncensored_only",
                    "train_cases_total": len(train_joint_cases),
                    "train_uncensored_cases_n": len(train_u_cases),
                    "train_censored_cases_n": len(train_c_cases),
                    "test_cases_total": len(test_u_cases),
                    "test_filter": "censored_removed_from_test",
                    "notes": "旧设计，对照用",
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--official_splits_root",
        type=Path,
        default=_ROOT / "data" / "splits" / "LUAD",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        default=_ROOT / "data" / "splits" / "censored_stage_protocol_LUAD",
    )
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--n_outer", type=int, default=5, help="Phase-1 官折外 5 折（继承 k=0..4）")
    p.add_argument("--n_fold_phase2", type=int, default=5, help="Phase-2 独立新分的 5 大折（用户说的 5 轮）")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    args.official_splits_root = args.official_splits_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    bundles = build_phase1_outer_splits(
        args.official_splits_root,
        args.out_root,
        args.target_col,
        args.n_outer,
        args.seed,
    )
    build_phase2_independent_5fold_splits(
        args.official_splits_root,
        args.out_root,
        args.target_col,
        int(args.n_fold_phase2),
        args.seed,
    )
    print(f"[OK] splits written to: {args.out_root}")
    print(f"  phase1 outer5 (official k->uncensored only): {args.out_root / 'phase1_outer5'}")
    print(f"  phase2 independent5 (重新分5折，uncensored 5折 + censored 5份一一对应): {args.out_root / 'phase2_independent5'}")
    print(f"  phase2 每折有两套 eval 口径：eval_with_censored / eval_uncensored_only （用户说“都试试看吧”）")


if __name__ == "__main__":
    main()
