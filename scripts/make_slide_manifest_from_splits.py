from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _collect_slides_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "slide_id" not in df.columns:
        raise ValueError(f"missing slide_id column in: {path}")
    slides = [str(x).strip() for x in df["slide_id"].tolist() if str(x).strip()]
    return slides


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--splits_root", default="data/splits")
    p.add_argument("--protocol_prefix", default="censored_stage_protocol_")
    p.add_argument("--phase", default="phase2_independent5", choices=["phase1_outer5", "phase2_independent5"])
    p.add_argument("--leaf", default="eval_with_censored")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cohort = str(args.cohort).upper()
    folds = [int(x) for x in str(args.folds).split(",") if str(x).strip() != ""]
    splits_root = Path(str(args.splits_root)).resolve()
    base = splits_root / f"{str(args.protocol_prefix)}{cohort}" / str(args.phase)

    slides: set[str] = set()
    for k in folds:
        if str(args.phase) == "phase1_outer5":
            leaf_dir = base / f"fold_{k}"
        else:
            leaf_dir = base / f"fold_{k}" / str(args.leaf)
        for name in ("train.csv", "test.csv"):
            csv_path = leaf_dir / name
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            for s in _collect_slides_from_csv(csv_path):
                slides.add(s)

    out_path = Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(slides)) + "\n", encoding="utf-8")
    print(f"wrote {len(slides)} slides to {out_path}")


if __name__ == "__main__":
    main()

