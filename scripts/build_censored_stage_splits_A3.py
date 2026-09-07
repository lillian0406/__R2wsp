#!/usr/bin/env python3
"""
Build censored_stage_protocol splits for BRCA (and LUSC/BLCA when RNA ready)

Reads data/splits/<COHORT>/k={0..4}/{train,test}.csv (A3 format: case_id_submitter,
slide_stem_no_uuid, days_to_event, vital_status), merges with GDC-patched clinical.csv
to get LUAD-official style 60+ columns, then runs build_phase1_outer5 +
build_phase2_independent_5fold_splits exactly as in build_censored_stage_splits.py.

Cohort is hardcoded BRCA for now. Change --cohort later for LUSC/BLCA.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Reuse the helpers from build_censored_stage_splits.py
from build_censored_stage_splits import (
    stratified_kfold_case_ids,
    rows_from_case_ids,
    deduplicate_cases,
    filter_uncensored,
    filter_censored,
    write_csv,
    SplitBundle,
)

LUAD_REFERENCE_COLUMNS = [
    "case_id", "slide_id", "submitter_id",
    "tumor_type", "project_id", "site_of_resection_or_biopsy", "sex",
    "OncotreeCode", "cancer_type_detailed", "tissue_source_site", "is_code_valid",
    "mpp", "op", "level0_mag", "OncoTreeSiteCode",
    "age_at_initial_pathologic_diagnosis", "race", "ajcc_pathologic_tumor_stage",
    "clinical_stage", "histological_type", "histological_grade",
    "initial_pathologic_dx_year", "menopause_status", "birth_days_to",
    "vital_status", "tumor_status", "last_contact_days_to", "death_days_to",
    "cause_of_death",
    "new_tumor_event_type", "new_tumor_event_site", "new_tumor_event_site_other",
    "new_tumor_event_dx_days_to", "treatment_outcome_first_course",
    "margin_status", "residual_tumor",
    "os_censorship", "os_survival_days",
    "dss_censorship", "dss_survival_days",
    "dfi_censorship", "dfi_survival_days",
    "pfi_censorship", "pfi_survival_days",
    "redaction",
    "pfi_v1_censorship", "pfi_v1_survival_days",
    "pfi_v2_censorship", "pfi_v2_survival_days",
    "pfs_censorship", "pfs_survival_days",
    "dss_cr_censorship", "dss_cr_survival_days",
    "dfi_cr_censorship", "dfi_cr_survival_days",
    "pfi_cr_censorship", "pfi_cr_survival_days",
]

AUDIT_SVS = _ROOT / "outputs" / "_disk_cohort_inventory" / "svs_5063_cohort_audit_TCGA_OFFICIAL_FINAL.csv"

SHORT_SLIDE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z]-[A-Z0-9]{2}-DX[A-Z0-9])", re.IGNORECASE)
H5_GLOB_DIR = _ROOT / "data" / "wsi_features" / "extracted_mag20x_patch256_fp" / "uni1024" / "feats_h5"


def _find_h5_for_slide_stem(slide_stem_no_uuid: str, all_h5_names: dict[str, list[str]]) -> str | None:
    """Return full H5 stem (including UUID suffix) matching the short slide stem."""
    return all_h5_names.get(slide_stem_no_uuid.upper(), [None])[0]


def load_A3_rows(cohort: str, k: int, all_h5_names: dict[str, list[str]], clinical_df: pd.DataFrame) -> list[dict]:
    """Load A3 k-fold CSV, enrich with clinical data, return LUAD-style rows."""
    a3 = _ROOT / "data" / "splits" / cohort / f"k={k}"
    out_rows = []
    for tag in ("train", "test"):
        fp = a3 / f"{tag}.csv"
        rows = list(csv.DictReader(open(fp)))
        for r in rows:
            sub = r["case_id_submitter"].upper()
            slide_stem = r["slide_stem_no_uuid"].upper()
            dte = int(r["days_to_event"])
            vs_code = int(r["vital_status"])  # 1=Dead, 0=Alive
            vs_text = "Dead" if vs_code == 1 else "Alive"
            # Find full slide_id (h5 filename stem with uuid)
            full_slide_id = _find_h5_for_slide_stem(slide_stem, all_h5_names) or slide_stem
            # Build LUAD-style row: fill all 60+ cols, fill known ones, rest empty
            row = {c: "" for c in LUAD_REFERENCE_COLUMNS}
            row["case_id"] = sub
            row["submitter_id"] = sub
            row["slide_id"] = full_slide_id
            row["project_id"] = f"TCGA-{cohort}"
            row["vital_status"] = vs_text
            # Clinical merge
            clin_match = clinical_df[clinical_df["submitter_id"] == sub]
            if len(clin_match) > 0:
                cr = clin_match.iloc[0].to_dict()
                for col in ("ajcc_pathologic_stage", "ajcc_pathologic_t", "tumor_stage", "grade",
                            "age_at_initial_pathologic_diagnosis", "sex", "race",
                            "ajcc_pathologic_tumor_stage", "histological_grade",
                            "initial_pathologic_dx_year", "histological_type"):
                    if col in cr and cr[col] != "" and cr[col] is not None:
                        row[col] = cr[col]
                if pd.notna(cr.get("os_censorship", None)) and str(cr.get("os_censorship", "")) != "":
                    row["os_censorship"] = float(cr["os_censorship"])
                    row["os_survival_days"] = float(cr["os_survival_days"])
                    row["dss_censorship"] = float(cr["dss_censorship"])
                    row["dss_survival_days"] = float(cr["dss_survival_days"])
                    if vs_code == 1:
                        row["death_days_to"] = float(cr["dss_survival_days"])
                        row["last_contact_days_to"] = ""
                    else:
                        row["last_contact_days_to"] = float(cr["dss_survival_days"])
                        row["death_days_to"] = ""
            # If clinical merge failed, fall back to days_to_event/vital_status from A3
            if row["os_censorship"] == "" or row["os_censorship"] is None:
                censorship = 0.0 if vs_code == 1 else 1.0
                event_time = float(dte)
                row["os_censorship"] = censorship
                row["os_survival_days"] = event_time
                row["dss_censorship"] = censorship
                row["dss_survival_days"] = event_time
                if vs_code == 1:
                    row["death_days_to"] = event_time
                else:
                    row["last_contact_days_to"] = event_time
            out_rows.append(row)
    return out_rows


def build_all_k_rows(cohort: str, clinical_df: pd.DataFrame) -> dict[int, list[dict]]:
    """Load k=0..4 A3 rows and return dict[k] -> train+test combined rows."""
    # Index all h5 files by short slide stem
    all_h5_names: dict[str, list[str]] = defaultdict(list)
    for f in sorted(H5_GLOB_DIR.glob("*.h5")):
        m = SHORT_SLIDE_RE.match(f.name)
        if m:
            all_h5_names[m.group(1).upper()].append(f.stem)
    print(f"[index] h5 files in uni1024/feats_h5: {len(list(H5_GLOB_DIR.glob('*.h5')))}, unique short stems: {len(all_h5_names)}")
    k_rows = {}
    for k in range(5):
        rows = load_A3_rows(cohort, k, all_h5_names, clinical_df)
        train_n = sum(1 for r in rows if r["case_id"] in set(
            rr["case_id_submitter"].upper() for rr in csv.DictReader(open(_ROOT / "data" / "splits" / cohort / f"k={k}" / "train.csv"))
        ))
        # Actually easier: re-load
    return load_k_rows_with_split_tags(cohort, all_h5_names, clinical_df)


def load_k_rows_with_split_tags(cohort: str, all_h5_names: dict[str, list[str]], clinical_df: pd.DataFrame):
    result = {}
    for k in range(5):
        k_train = []
        k_test = []
        for tag, out_list in (("train", k_train), ("test", k_test)):
            fp = _ROOT / "data" / "splits" / cohort / f"k={k}" / f"{tag}.csv"
            for r in csv.DictReader(open(fp)):
                sub = r["case_id_submitter"].upper()
                slide_stem = r["slide_stem_no_uuid"].upper()
                dte = int(r["days_to_event"])
                vs_code = int(r["vital_status"])
                vs_text = "Dead" if vs_code == 1 else "Alive"
                full_slide_id = _find_h5_for_slide_stem(slide_stem, all_h5_names) or slide_stem
                row = {c: "" for c in LUAD_REFERENCE_COLUMNS}
                row["case_id"] = sub
                row["submitter_id"] = sub
                row["slide_id"] = full_slide_id
                row["project_id"] = f"TCGA-{cohort}"
                row["vital_status"] = vs_text
                clin_match = clinical_df[clinical_df["submitter_id"] == sub]
                if len(clin_match) > 0:
                    cr = clin_match.iloc[0].to_dict()
                    for col in ("ajcc_pathologic_stage", "ajcc_pathologic_t", "tumor_stage", "grade",
                                "age_at_initial_pathologic_diagnosis", "sex", "race",
                                "ajcc_pathologic_tumor_stage", "histological_grade",
                                "initial_pathologic_dx_year", "histological_type"):
                        if col in cr and cr[col] != "" and cr[col] is not None and not (isinstance(cr[col], float) and np.isnan(cr[col])):
                            row[col] = cr[col]
                    censor_val = cr.get("os_censorship", "")
                    try:
                        if censor_val != "" and censor_val is not None and not pd.isna(censor_val):
                            censorship = float(censor_val)
                            event_time = float(cr["os_survival_days"])
                            row["os_censorship"] = censorship
                            row["os_survival_days"] = event_time
                            row["dss_censorship"] = float(cr["dss_censorship"])
                            row["dss_survival_days"] = float(cr["dss_survival_days"])
                            if abs(censorship - 0.0) < 0.5:
                                row["death_days_to"] = event_time
                            else:
                                row["last_contact_days_to"] = event_time
                    except (ValueError, TypeError, KeyError):
                        pass
                if row["os_censorship"] == "":
                    censorship = 0.0 if vs_code == 1 else 1.0
                    event_time = float(dte)
                    row["os_censorship"] = censorship
                    row["os_survival_days"] = event_time
                    row["dss_censorship"] = censorship
                    row["dss_survival_days"] = event_time
                    if vs_code == 1:
                        row["death_days_to"] = event_time
                    else:
                        row["last_contact_days_to"] = event_time
                out_list.append(row)
        result[k] = (k_train, k_test)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Enforce LUAD_REFERENCE_COLUMNS order with empty fill for missing cols
    for c in LUAD_REFERENCE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[LUAD_REFERENCE_COLUMNS]
    df.to_csv(path, index=False)
    return


def build_phase1_outer_splits(k_rows: dict, out_root: Path, target_col: str, n_outer: int, seed: int, cohort: str = "BRCA") -> list[SplitBundle]:
    bundles = []
    for k in range(n_outer):
        train_all, test_all = k_rows[k]
        train_u = filter_uncensored(train_all, target_col)
        test_u = filter_uncensored(test_all, target_col)
        out_dir = out_root / "phase1_outer5" / f"fold_{k}"
        write_csv(out_dir / "train.csv", train_u)
        write_csv(out_dir / "test.csv", test_u)
        manifest = {
            "cohort": cohort,
            "official_fold": k,
            "target_col": target_col,
            "filter": "uncensored_only_train_val_test",
            "n_train_cases_all": len(set(r["case_id"] for r in train_all)),
            "n_test_cases_all": len(set(r["case_id"] for r in test_all)),
            "n_train_cases_uncensored": len(set(r["case_id"] for r in train_u)),
            "n_test_cases_uncensored": len(set(r["case_id"] for r in test_u)),
            "seed": seed,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        bundles.append(SplitBundle(out_dir, out_dir / "train.csv", out_dir / "test.csv", manifest))
        print(f"  P1 fold_{k}: train_u={manifest['n_train_cases_uncensored']}, test_u={manifest['n_test_cases_uncensored']}")
    return bundles


def build_phase2_independent_5fold_splits(k_rows: dict, out_root: Path, target_col: str, n_folds: int, seed: int, cohort: str = "BRCA"):
    # Union global pool from all k (dedup)
    all_rows_map: dict[str, dict] = {}
    for k in range(5):
        for r in k_rows[k][0] + k_rows[k][1]:
            cid = str(r["case_id"])
            if cid not in all_rows_map:
                all_rows_map[cid] = r
    all_rows = list(all_rows_map.values())
    uncensored_rows = filter_uncensored(all_rows, target_col)
    censored_rows = filter_censored(all_rows, target_col)
    print(f"  P2 global: cases_total={len(all_rows)}, uncensored={len(uncensored_rows)}, censored={len(censored_rows)}")
    u_folds = stratified_kfold_case_ids(
        deduplicate_cases(uncensored_rows), target_col=target_col, n_folds=n_folds, seed=seed
    )
    c_folds = stratified_kfold_case_ids(
        deduplicate_cases(censored_rows), target_col=target_col, n_folds=n_folds, seed=seed + 9999
    )
    top_root = out_root / "phase2_independent5"
    top_root.mkdir(parents=True, exist_ok=True)
    (top_root / "manifest.json").write_text(
        json.dumps(
            {
                "cohort": cohort,
                "target_col": target_col,
                "n_folds": int(n_folds),
                "split_kind": "independent5_not_inherited_from_official",
                "uncensored_total_cases": len({r["case_id"] for r in uncensored_rows}),
                "censored_total_cases": len({r["case_id"] for r in censored_rows}),
                "uncensored_folds": [sorted(f) for f in u_folds],
                "censored_folds": [sorted(f) for f in c_folds],
                "per_fold_case_counts": [
                    {"fold": int(i), "uncensored": len(u_folds[i]), "censored": len(c_folds[i])}
                    for i in range(n_folds)
                ],
                "seed": seed,
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
        test_with_censor_cases = test_u_cases | test_c_cases
        train_rows = [all_rows_map[cid] for cid in sorted(train_joint_cases)]
        test_with_cens_rows = [all_rows_map[cid] for cid in sorted(test_with_censor_cases)]
        rootA = top_root / f"fold_{i}" / "eval_with_censored"
        write_csv(rootA / "train.csv", train_rows)
        write_csv(rootA / "test.csv", test_with_cens_rows)
        (rootA / "manifest.json").write_text(
            json.dumps(
                {
                    "fold": int(i), "eval_kind": "eval_with_censored",
                    "train_cases_total": len(train_joint_cases),
                    "train_uncensored_cases_n": len(train_u_cases),
                    "train_censored_cases_n": len(train_c_cases),
                    "test_cases_total": len(test_with_censor_cases),
                    "test_uncensored_cases_n": len(test_u_cases),
                    "test_censored_cases_n": len(test_c_cases),
                    "notes": "censored_stage_protocol 两阶段 P2 默认口径",
                },
                indent=2, ensure_ascii=False,
            )
        )
        test_uncens_only_rows = [all_rows_map[cid] for cid in sorted(test_u_cases)]
        rootB = top_root / f"fold_{i}" / "eval_uncensored_only"
        write_csv(rootB / "train.csv", train_rows)
        write_csv(rootB / "test.csv", test_uncens_only_rows)
        (rootB / "manifest.json").write_text(
            json.dumps(
                {
                    "fold": int(i), "eval_kind": "eval_uncensored_only",
                    "train_cases_total": len(train_joint_cases),
                    "train_uncensored_cases_n": len(train_u_cases),
                    "train_censored_cases_n": len(train_c_cases),
                    "test_cases_total": len(test_u_cases),
                    "test_filter": "censored_removed_from_test",
                    "notes": "Δ = uncensored - with_censored 删失偏倚量化指标对照口径",
                },
                indent=2, ensure_ascii=False,
            )
        )
        print(f"  P2 fold_{i}: train(U+C)={len(train_joint_cases)}, test_with_cens={len(test_with_censor_cases)}, test_uncens_only={len(test_u_cases)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA", choices=["BRCA", "LUSC", "BLCA", "LUAD", "PAAD","THCA"])
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--target_col", default="dss_survival_days")
    ap.add_argument("--n_outer", type=int, default=5)
    ap.add_argument("--n_fold_phase2", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = _ROOT / "data" / "splits" / f"censored_stage_protocol_{args.cohort}"

    print(f"=== censored_stage_protocol for {args.cohort} (seed={args.seed}) ===")
    # 1. Load clinical
    clin_p = _ROOT / "data" / "raw_clinical" / args.cohort / "clinical.csv"
    clinical_df = pd.read_csv(clin_p)
    clinical_df["submitter_id"] = clinical_df["submitter_id"].str.upper()
    print(f"  Clinical rows: {len(clinical_df)}")

    # 2. Index h5 + load A3 k rows
    all_h5_names: dict[str, list[str]] = defaultdict(list)
    for f in sorted(H5_GLOB_DIR.glob("*.h5")):
        m = SHORT_SLIDE_RE.match(f.name)
        if m:
            all_h5_names[m.group(1).upper()].append(f.stem)
    print(f"  h5 indexed: {len(all_h5_names)} short stems")
    k_rows = load_k_rows_with_split_tags(args.cohort, all_h5_names, clinical_df)
    n_known = sum(len(t[0]) + len(t[1]) for t in k_rows.values())
    print(f"  Total rows loaded from A3 splits: {n_known} (slides across k=0..4)")

    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # 3. Build P1 + P2
    print("\n[Phase-1 outer5 (uncensored only, inherits A3 k-fold split assignment)]")
    build_phase1_outer_splits(k_rows, args.out_root, args.target_col, args.n_outer, args.seed, cohort=args.cohort)
    print("\n[Phase-2 independent5 (global pool re-5fold, uncensored+censored stratified)]")
    build_phase2_independent_5fold_splits(k_rows, args.out_root, args.target_col, int(args.n_fold_phase2), args.seed, cohort=args.cohort)

    print(f"\n[OK] censored_stage_protocol_{args.cohort} written → {args.out_root}")
    print(f"  phase1_outer5 (official k→uncensored only): {args.out_root / 'phase1_outer5'}")
    print(f"  phase2_independent5 (重新分5折，uncensored 5折 + censored 5份一一对应): {args.out_root / 'phase2_independent5'}")
    print(f"  phase2 每折有两套 eval 口径：eval_with_censored / eval_uncensored_only （Δ指标对照用）")


if __name__ == "__main__":
    main()
