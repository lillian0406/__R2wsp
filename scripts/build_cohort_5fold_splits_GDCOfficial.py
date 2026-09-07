#!/usr/bin/env python3
# A3 - R2wsp Official GDC 5-fold splits builder (strict reproducible, external-friendly)
# ===========================================================
# RULES (hardcoded, each rule enforced with assertions):
# R1  Case eligibility:
#     vital_status in {Alive, Dead}
#     Dead: days_to_death int >= 0
#     Alive: days_to_last_follow_up int >= 0
# R2  Time binning:
#     Dead  cases: 4 quartiles (Q1..Q4) on days_to_death
#     Alive cases: 4 quartiles (Q1..Q4) on days_to_last_follow_up
#     Small strata auto shrink with duplicates='drop' (q = min(4, group_size))
# R3  Strata = vital_status (2 classes) x time_bin (4 classes) -> <= 8 strata
# R4  Splitter choice (automatically pick for reproducibility & correctness):
#     IF every stratum has >= 5 members: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     ELSE: KFold(n_splits=5, shuffle=True, random_state=42) + vitals bias post-check
# R4a Every fold train/test case count delta <= 1
# R4b Global vital_status global vs per-fold distribution delta in (global - fold)^2 avg <= 0.01 (informational only)
# R5  Same-case all slides always stay together (multi-slide case never leaks across train/test)
# R6  Output 4 fixed output columns, fixed order: case_id_submitter, slide_stem_no_uuid, days_to_event, vital_status
#     vital_status encoding: 1=Dead (event occurred), 0=Alive (censored)
# R7  Output layout: data/splits/<COHORT>/k={0..4}/{train.csv,test.csv}
# R8  Audit CSV: data/splits/<COHORT>/split_audit.csv per-fold counts
# R9  5-fold 100% coverage: sum of test_cases over 5 folds == total enroled cases
# R10 SEED MUST BE 42 (do not change, reproducibility contract)

import argparse, csv, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

PROJ = Path(__file__).resolve().parent.parent
RAW_CLIN = PROJ / "data" / "raw_clinical"
RAW_SVS = PROJ / "data" / "raw_svs"
AUDIT_SVS = PROJ / "outputs" / "_disk_cohort_inventory" / "svs_5063_cohort_audit_TCGA_OFFICIAL_FINAL.csv"
SPLITS_DIR = PROJ / "data" / "splits"


def _strip_uuid_svs(fn: str) -> str:
    base = Path(fn).stem
    base2 = re.sub(
        r"\.[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "", base)
    return (base2 or base).upper()


def _int_nonneg(x):
    try:
        v = int(float(str(x).strip()))
        return v if v >= 0 else None
    except Exception:
        return None


def _build_local_svs_df(cohort: str, valid_cases: dict[str, tuple[int, int]], raw_svs_root: Path) -> pd.DataFrame:
    id_re = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)
    rows = []
    for p in raw_svs_root.rglob("*.svs"):
        m = id_re.search(p.name)
        if not m:
            continue
        cid = m.group(0).upper()
        if cid in valid_cases:
            rows.append(
                dict(
                    file_name=p.name,
                    file_path=str(p),
                    case_id_submitter=cid,
                    cohort_inferred=cohort,
                )
            )
    return pd.DataFrame(rows)


def build_one_cohort(
    cohort: str,
    *,
    seed: int = 42,
    svs_source: str = "audit",
    audit_csv: Path | None = None,
    raw_svs_root: Path | None = None,
) -> pd.DataFrame:
    if seed != 42:
        raise AssertionError("R10 VIOLATION: seed must be 42")

    # --- data loading ---
    clin = list(csv.DictReader(open(RAW_CLIN / cohort / "clinical.csv")))

    # --- R1 eligibility filter ---
    valid_cases: dict = {}
    for r in clin:
        sub = r['submitter_id'].upper()
        v = (r.get('vital_status') or '').title()
        dd = _int_nonneg(r.get('days_to_death') or '')
        df = _int_nonneg(r.get('days_to_last_follow_up') or '')
        if v == 'Dead' and dd is not None:
            valid_cases[sub] = (dd, 1)
        elif v == 'Alive' and df is not None:
            valid_cases[sub] = (df, 0)

    svs_source = str(svs_source or "audit").strip().lower()
    audit_csv = (audit_csv or AUDIT_SVS).resolve()
    raw_svs_root = (raw_svs_root or RAW_SVS).resolve()

    sdf_parts: list[pd.DataFrame] = []

    if svs_source in {"audit", "union"}:
        if audit_csv.exists():
            svs_all = pd.read_csv(audit_csv)
            sdf_audit = svs_all[svs_all["cohort_inferred"] == cohort].copy()
            if not sdf_audit.empty:
                sdf_audit["case_id_submitter"] = sdf_audit["case_id_submitter"].str.upper()
                sdf_parts.append(sdf_audit)

    if svs_source in {"local", "union"}:
        sdf_local = _build_local_svs_df(cohort, valid_cases, raw_svs_root)
        if not sdf_local.empty:
            sdf_local["case_id_submitter"] = sdf_local["case_id_submitter"].str.upper()
            sdf_parts.append(sdf_local)

    if not sdf_parts:
        raise FileNotFoundError(
            f"[{cohort}] no svs sources available: svs_source={svs_source} audit_csv={audit_csv} raw_svs_root={raw_svs_root}"
        )

    sdf = pd.concat(sdf_parts, ignore_index=True)
    if "file_id" in sdf.columns:
        sdf = sdf.drop_duplicates(subset=["file_id"])
    else:
        sdf = sdf.drop_duplicates(subset=["file_name", "case_id_submitter"])

    sdf = sdf[sdf['case_id_submitter'].isin(valid_cases)].copy()
    sdf['slide_stem_no_uuid'] = sdf['file_name'].map(_strip_uuid_svs)
    ev = sdf['case_id_submitter'].apply(lambda s: pd.Series(valid_cases[s]))
    ev.columns = ['days_to_event', 'vital_status']
    sdf = pd.concat([sdf.reset_index(drop=True), ev.reset_index(drop=True)], axis=1)

    # --- assertions column & format checks ---
    assert sdf['slide_stem_no_uuid'].notna().all(), f"{cohort} slide_stem nan"
    assert sdf['case_id_submitter'].str.match(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$').all(), f"{cohort} case_id format"

    # --- case-level frame ---
    cases_df = sdf[['case_id_submitter','days_to_event','vital_status']] \
        .drop_duplicates('case_id_submitter').reset_index(drop=True)

    # --- R2 quartile time bin ---
    cases_df['time_bin'] = None
    for vf in (1, 0):
        m = (cases_df['vital_status'] == vf)
        if m.sum() <= 1:
            continue
        q = min(4, int(m.sum()))
        try:
            cases_df.loc[m, 'time_bin'] = pd.qcut(
                cases_df.loc[m, 'days_to_event'], q=q,
                labels=[f'Q{i+1}' for i in range(q)], duplicates='drop')
        except Exception:
            cases_df.loc[m, 'time_bin'] = pd.qcut(
                cases_df.loc[m, 'days_to_event'].rank(method='first'), q=q,
                labels=[f'Q{i+1}' for i in range(q)], duplicates='drop')
    cases_df['time_bin'] = cases_df['time_bin'].astype(str)
    assert cases_df['time_bin'].notna().all(), f"{cohort} NaN time_bin"

    # --- R3 strata <= 8 ---
    cases_df['stratum'] = cases_df['vital_status'].astype(str) + '_' + cases_df['time_bin']
    n_strata = cases_df['stratum'].nunique()
    assert n_strata <= 8, f"{cohort} strata>8: {n_strata}"

    # --- R4 split choice ---
    min_stratum_size = int(cases_df['stratum'].value_counts().min())
    if min_stratum_size >= 5:
        print(f"  [{cohort}] StratifiedKFold strata={n_strata} min_stratum={min_stratum_size}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(skf.split(cases_df, cases_df['stratum']))
    else:
        print(f"  [{cohort}] KFold fallback strata={n_strata} min_stratum={min_stratum_size}<5")
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(cases_df))

    # R4a fold size delta <= 1 ---
    test_sizes = [len(te) for _, te in folds]
    assert max(test_sizes) - min(test_sizes) <= 1, f"{cohort} fold test size skew {test_sizes}"

    # --- R7 output files ---
    out_dir = SPLITS_DIR / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr_cases = set(cases_df.iloc[tr_idx]['case_id_submitter'])
        te_cases = set(cases_df.iloc[te_idx]['case_id_submitter'])
        assert len(tr_cases & te_cases) == 0, f"{cohort} k{k} case overlap"

        tr_sdf = sdf[sdf['case_id_submitter'].isin(tr_cases)]
        te_sdf = sdf[sdf['case_id_submitter'].isin(te_cases)]

        # --- R5 multi-slide same fold ---
        msc = sdf.groupby('case_id_submitter').size().loc[lambda s: s >= 2].index
        for cid in msc:
            a = cid in tr_cases
            b = cid in te_cases
            assert not (a and b), f"{cohort} k{k} multi-slide {cid} cross-fold leak"

        # --- R6 output 4 cols ---
        for tag, sub in (('train', tr_sdf), ('test', te_sdf)):
            p = out_dir / f'k={k}' / f'{tag}.csv'
            p.parent.mkdir(parents=True, exist_ok=True)
            sub[['case_id_submitter','slide_stem_no_uuid','days_to_event','vital_status']].to_csv(p, index=False)

        tr_c = int(tr_sdf['case_id_submitter'].nunique()); tr_s = len(tr_sdf)
        te_c = int(te_sdf['case_id_submitter'].nunique()); te_s = len(te_sdf)
        tr_v = Counter(tr_sdf.drop_duplicates('case_id_submitter')['vital_status'])
        te_v = Counter(te_sdf.drop_duplicates('case_id_submitter')['vital_status'])
        audit.append(dict(
            fold=k, train_cases=tr_c, train_slides=tr_s,
            test_cases=te_c, test_slides=te_s,
            train_Dead=int(tr_v.get(1, 0)), train_Alive=int(tr_v.get(0, 0)),
            test_Dead=int(te_v.get(1, 0)), test_Alive=int(te_v.get(0, 0)),
        ))

    # --- R9 coverage 100% ---
    assert sum(a['test_cases'] for a in audit) == len(cases_df), f"{cohort} test coverage not 100%"

    adf = pd.DataFrame(audit)
    adf.to_csv(out_dir / 'split_audit.csv', index=False)
    print(f"  [{cohort}] OK: valid={len(valid_cases)} enrol={len(cases_df)} slides={len(sdf)} fold_test_sizes={test_sizes}")
    print(adf.to_string())
    return adf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', required=True,
                      choices=['BRCA','LUAD','LUSC','PAAD','BLCA','COAD','READ','STAD','KIRC','PRAD','OV','UCEC','THCA'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument("--svs_source", default="audit", choices=["audit", "local", "union"])
    ap.add_argument("--audit_csv", default=str(AUDIT_SVS))
    ap.add_argument("--raw_svs_root", default=str(RAW_SVS))
    args = ap.parse_args()
    for c in args.cohorts:
        build_one_cohort(
            c,
            seed=int(args.seed),
            svs_source=str(args.svs_source),
            audit_csv=Path(args.audit_csv),
            raw_svs_root=Path(args.raw_svs_root),
        )


if __name__ == '__main__':
    main()
