#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _load_mmp_dss_table(mmp_survival_dir: Path, cohort: str) -> pd.DataFrame:
    rows = []
    for p in sorted(mmp_survival_dir.glob(f"TCGA_{cohort}_overall_survival_k=*/train.csv")):
        rows.append(pd.read_csv(p, usecols=["case_id", "dss_survival_days", "dss_censorship"]))
    for p in sorted(mmp_survival_dir.glob(f"TCGA_{cohort}_overall_survival_k=*/test.csv")):
        rows.append(pd.read_csv(p, usecols=["case_id", "dss_survival_days", "dss_censorship"]))
    if not rows:
        raise FileNotFoundError(f"no MMP survival splits found for cohort={cohort} under {mmp_survival_dir}")
    df = pd.concat(rows, ignore_index=True)
    df["case_id"] = df["case_id"].astype(str).str.upper()
    df = df.drop_duplicates("case_id")
    df["dss_survival_days"] = pd.to_numeric(df["dss_survival_days"], errors="coerce")
    df["dss_censorship"] = pd.to_numeric(df["dss_censorship"], errors="coerce")
    df = df.dropna(subset=["dss_survival_days", "dss_censorship"]).copy()
    df = df[(df["dss_survival_days"] >= 0) & (df["dss_censorship"].isin([0.0, 1.0]))].copy()
    return df


def _audit_one_split(df: pd.DataFrame) -> dict[str, int]:
    cases = int(df["case_id_submitter"].nunique())
    slides = int(len(df))
    vc = Counter(df.drop_duplicates("case_id_submitter")["vital_status"].astype(int).tolist())
    return {
        "cases": cases,
        "slides": slides,
        "event": int(vc.get(1, 0)),
        "censored": int(vc.get(0, 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--in_a3_dir", type=Path, default=None)
    ap.add_argument("--out_cohort", required=True)
    ap.add_argument("--mmp_survival_dir", type=Path, default=Path("/root/autodl-tmp/_refs/MMP-main/src/splits/survival"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cohort = str(args.cohort).strip().upper()
    in_a3_dir = (args.in_a3_dir or Path("/root/autodl-tmp/R2wsp/data/splits") / cohort).resolve()
    out_dir = (Path("/root/autodl-tmp/R2wsp/data/splits") / str(args.out_cohort).strip()).resolve()

    mmp = _load_mmp_dss_table(args.mmp_survival_dir.resolve(), cohort).set_index("case_id")
    mmp_days = mmp["dss_survival_days"].astype(float)
    mmp_event = (1.0 - mmp["dss_censorship"].astype(float)).astype(int)

    changed_event_cases = set()
    changed_days_cases = set()
    touched_cases = set()
    missing_cases = set()

    audit_rows = []
    for k in range(5):
        for tag in ("train", "test"):
            p = in_a3_dir / f"k={k}" / f"{tag}.csv"
            df = pd.read_csv(p)
            df["case_id_submitter"] = df["case_id_submitter"].astype(str).str.upper()
            df["days_to_event"] = pd.to_numeric(df["days_to_event"], errors="coerce")
            df["vital_status"] = pd.to_numeric(df["vital_status"], errors="coerce")
            if df["days_to_event"].isna().any() or df["vital_status"].isna().any():
                raise ValueError(f"invalid numeric columns in {p}")

            before_case = df[["case_id_submitter", "days_to_event", "vital_status"]].drop_duplicates("case_id_submitter").set_index("case_id_submitter")
            before_cases = set(before_case.index.tolist())

            mapped = before_case.index.intersection(mmp.index)
            if args.strict:
                keep = set(mapped.tolist())
                df = df[df["case_id_submitter"].isin(keep)].copy()
            else:
                keep = set(before_cases)

            after_case = df[["case_id_submitter", "days_to_event", "vital_status"]].drop_duplicates("case_id_submitter").set_index("case_id_submitter")
            after_cases = set(after_case.index.tolist())

            for cid in sorted(after_case.index.intersection(mmp.index)):
                touched_cases.add(cid)
                new_days = float(mmp_days.loc[cid])
                new_event = int(mmp_event.loc[cid])
                old_days = float(after_case.loc[cid, "days_to_event"])
                old_event = int(after_case.loc[cid, "vital_status"])
                if old_event != new_event:
                    changed_event_cases.add(cid)
                if old_days != new_days:
                    changed_days_cases.add(cid)
                df.loc[df["case_id_submitter"] == cid, "days_to_event"] = new_days
                df.loc[df["case_id_submitter"] == cid, "vital_status"] = new_event

            if not args.strict:
                missing = after_case.index.difference(mmp.index)
                missing_cases.update(missing.tolist())

            out_p = out_dir / f"k={k}" / f"{tag}.csv"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)

        tr = pd.read_csv(out_dir / f"k={k}" / "train.csv")
        te = pd.read_csv(out_dir / f"k={k}" / "test.csv")
        a_tr = _audit_one_split(tr)
        a_te = _audit_one_split(te)
        audit_rows.append(
            {
                "fold": k,
                "train_cases": a_tr["cases"],
                "train_slides": a_tr["slides"],
                "test_cases": a_te["cases"],
                "test_slides": a_te["slides"],
                "train_event": a_tr["event"],
                "train_censored": a_tr["censored"],
                "test_event": a_te["event"],
                "test_censored": a_te["censored"],
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out_dir / "split_audit.csv", index=False)

    print("=== relabel A3 splits from MMP DSS ===")
    print("cohort", cohort)
    print("in_a3_dir", str(in_a3_dir))
    print("out_dir", str(out_dir))
    print("mmp_cases", int(len(mmp)))
    print("touched_cases", int(len(touched_cases)))
    print("missing_cases_kept", int(len(missing_cases)))
    print("changed_event_cases", int(len(changed_event_cases)))
    print("changed_days_cases", int(len(changed_days_cases)))
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    main()

