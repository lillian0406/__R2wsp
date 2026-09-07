from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TCGA_CASE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def _extract_case_id(s: str) -> str | None:
    m = TCGA_CASE_RE.search(str(s))
    return m.group(0).upper() if m else None


def _collect_existing_cases_from_dir(root: Path) -> set[str]:
    if not root.exists():
        return set()
    cases: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        cid = _extract_case_id(p.name)
        if cid:
            cases.add(cid)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--clinical_csv", default=None)
    ap.add_argument("--time_col", default="dss_survival_days")
    ap.add_argument("--censorship_col", default="dss_censorship")
    ap.add_argument("--case_col", default="submitter_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_existing", action="store_true")
    ap.add_argument("--raw_svs_dir", default=None)
    ap.add_argument("--h5_dir", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cohort = str(args.cohort).upper()
    clinical_csv = Path(args.clinical_csv) if args.clinical_csv else root / "data" / "raw_clinical" / cohort / "clinical.csv"

    df = pd.read_csv(clinical_csv)
    if str(args.case_col) not in df.columns:
        raise ValueError(f"case_col not in clinical.csv: {args.case_col}")
    if str(args.time_col) not in df.columns:
        raise ValueError(f"time_col not in clinical.csv: {args.time_col}")
    if str(args.censorship_col) not in df.columns:
        raise ValueError(f"censorship_col not in clinical.csv: {args.censorship_col}")

    case_ids = df[str(args.case_col)].astype(str).str.upper()
    t = pd.to_numeric(df[str(args.time_col)], errors="coerce")
    c = pd.to_numeric(df[str(args.censorship_col)], errors="coerce")

    ok = t.notna() & (t >= 0) & c.notna() & c.isin([0, 1])
    event = ok & (c == 0)
    cand = sorted(set(case_ids[event].tolist()))

    existing: set[str] = set()
    if bool(args.exclude_existing):
        raw_svs_dir = Path(args.raw_svs_dir) if args.raw_svs_dir else root / "data" / "raw_svs"
        h5_dir = Path(args.h5_dir) if args.h5_dir else root / "data" / "wsi_features" / "extracted_mag20x_patch256_fp" / "uni1024" / "feats_h5"
        existing |= _collect_existing_cases_from_dir(raw_svs_dir)
        existing |= _collect_existing_cases_from_dir(h5_dir)

    out_ids = [x for x in cand if x not in existing]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_ids) + ("\n" if out_ids else ""), encoding="utf-8")

    print(f"[clinical] rows={len(df)}")
    print(f"[event] candidate_cases={len(cand)}")
    if bool(args.exclude_existing):
        print(f"[local] existing_cases={len(existing)}")
    print(f"[out] cases={len(out_ids)} path={out}")


if __name__ == "__main__":
    main()

