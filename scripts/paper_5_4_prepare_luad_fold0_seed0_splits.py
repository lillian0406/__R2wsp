#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--target_col", type=str, default="dss_survival_days")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_uncensored", type=int, default=100)
    p.add_argument("--n_censored", type=int, default=100)
    args = p.parse_args()

    split_dir = Path(args.split_dir).resolve()
    out_root = Path(args.out_root).resolve()
    target_col = str(args.target_col)
    censorship_col = target_col.split("_")[0] + "_censorship"

    train_rows = _read_rows(split_dir / "train.csv")
    test_rows = _read_rows(split_dir / "test.csv")
    seen: set[str] = set()
    pool: list[dict[str, str]] = []
    for r in train_rows + test_rows:
        cid = str(r.get("case_id", ""))
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if target_col not in r or censorship_col not in r:
            continue
        pool.append(r)

    uncensored = [r for r in pool if float(r[censorship_col]) < 0.5]
    censored = [r for r in pool if float(r[censorship_col]) >= 0.5]

    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(censored)

    n_u_req = int(args.n_uncensored)
    n_c_req = int(args.n_censored)
    u_rows = list(uncensored[: min(n_u_req, len(uncensored))])
    c_rows = list(censored[: min(n_c_req, len(censored))])

    fieldnames = ["case_id", target_col, censorship_col]
    u_min = [{"case_id": str(r["case_id"]), target_col: str(r[target_col]), censorship_col: str(r[censorship_col])} for r in u_rows]
    c_min = [{"case_id": str(r["case_id"]), target_col: str(r[target_col]), censorship_col: str(r[censorship_col])} for r in c_rows]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_split_dir": str(split_dir),
                "target_col": target_col,
                "censorship_col": censorship_col,
                "seed": int(args.seed),
                "requested": {"n_uncensored": n_u_req, "n_censored": n_c_req},
                "available": {"n_uncensored": len(uncensored), "n_censored": len(censored)},
                "selected": {"n_uncensored": len(u_rows), "n_censored": len(c_rows)},
                "uncensored_case_ids": [str(r["case_id"]) for r in u_rows],
                "censored_case_ids": [str(r["case_id"]) for r in c_rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_u_only = out_root / "u_only"
    _write_rows(out_u_only / "train.csv", u_min, fieldnames)
    _write_rows(out_u_only / "test.csv", u_min, fieldnames)

    out_all_correct = out_root / "all_correct"
    _write_rows(out_all_correct / "train.csv", u_min + c_min, fieldnames)
    _write_rows(out_all_correct / "test.csv", u_min, fieldnames)

    out_all_masked = out_root / "all_masked"
    c_masked = [{**r, censorship_col: "0.0"} for r in c_min]
    _write_rows(out_all_masked / "train.csv", u_min + c_masked, fieldnames)
    _write_rows(out_all_masked / "test.csv", u_min, fieldnames)


if __name__ == "__main__":
    main()

