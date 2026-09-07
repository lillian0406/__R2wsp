#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t")
    idx = str(df.columns[0])
    df = df.set_index(idx)
    t = df.T
    if args.float32:
        t = t.astype(np.float32, copy=False)
    t.to_csv(out_path, sep="\t")


if __name__ == "__main__":
    main()

