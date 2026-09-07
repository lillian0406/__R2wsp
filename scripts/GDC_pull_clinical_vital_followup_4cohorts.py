#!/usr/bin/env python3
"""
A2 · Pull complete vital_status/days_to_death/days_to_last_follow_up from GDC /cases endpoint
for TCGA-BRCA, TCGA-LUSC, TCGA-BLCA, TCGA-LUAD (4 cohorts) and write LUAD-standard clinical.csv
with dss_censorship/dss_survival_days + os_censorship/os_survival_days columns.

Output: data/raw_clinical/<COHORT>/clinical.csv (LUAD-standard 60+ cols, empty fill for non-essential)

API docs: https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/#cases-fields
"""
import argparse, csv, json, os, time, sys
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

PROJ = Path(__file__).resolve().parent.parent
END = os.environ.get("GDC_API_BASE", "https://api.gdc.cancer.gov").rstrip("/")
OUT_CLIN_DIR = PROJ / "data" / "raw_clinical"

LUAD_REFERENCE_COLUMNS = [
    "submitter_id", "case_id", "project_id",
    "ajcc_pathologic_stage", "ajcc_pathologic_t", "tumor_stage", "grade",
    "vital_status", "days_to_death", "days_to_last_follow_up", "n_diagnoses",
    "age_at_initial_pathologic_diagnosis", "sex", "race", "ajcc_pathologic_tumor_stage",
    "clinical_stage", "histological_type", "histological_grade",
    "initial_pathologic_dx_year", "birth_days_to",
    "tumor_status", "last_contact_days_to", "death_days_to", "cause_of_death",
    "os_censorship", "os_survival_days",
    "dss_censorship", "dss_survival_days",
    "dfi_censorship", "dfi_survival_days",
    "pfi_censorship", "pfi_survival_days",
]

CASES_FIELDS = ",".join([
    "submitter_id", "case_id", "project.project_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.days_to_last_follow_up",
    "demographic.age_at_initial_pathologic_diagnosis",
    "demographic.gender",
    "demographic.race",
    "demographic.year_of_birth",
    "demographic.year_of_death",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.tumor_stage",
    "diagnoses.tumor_grade",
    "diagnoses.age_at_diagnosis",
    "diagnoses.year_of_diagnosis",
    "diagnoses.morphology",
    "diagnoses.primary_diagnosis",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.classification_of_tumor",
    "diagnoses.treatments.therapeutic_agents",
    "diagnoses.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "follow_ups.days_to_last_follow_up",
    "follow_ups.vital_status",
])


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def pull_cohort_cases(project_id: str, chunk_size: int = 200):
    """Pull ALL cases for a TCGA project (not limited by submitter list) via /cases endpoint."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": [project_id]}},
        ],
    }
    # First: count total cases
    params_count = {
        "filters": json.dumps(filters),
        "fields": "submitter_id",
        "format": "JSON",
        "size": 1,
    }
    r = requests.get(f"{END}/cases", params=params_count, timeout=60)
    r.raise_for_status()
    total = int(r.json()["data"]["pagination"]["total"])
    print(f"  [GDC] {project_id}: total {total} cases in GDC")

    # Paginate through all cases
    all_cases = []
    for start in range(0, total, chunk_size):
        params = {
            "filters": json.dumps(filters),
            "fields": CASES_FIELDS,
            "format": "JSON",
            "size": chunk_size,
            "from": start,
            "sort": "submitter_id:asc",
        }
        for retry in range(4):
            try:
                r = requests.get(f"{END}/cases", params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:
                wait = 2 ** retry
                print(f"  [WARN] retry {retry+1}/4 after {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"GDC /cases failed after 4 retries for {project_id} start={start}")
        hits = r.json()["data"]["hits"]
        for h in hits:
            all_cases.append(_flatten(h))
        print(f"    fetched {start + len(hits)}/{total}")
        time.sleep(0.3)
    return all_cases


def _flatten(h: dict) -> dict:
    demo = h.get("demographic", {}) or {}
    diagnoses = h.get("diagnoses") or [{}]
    d0 = diagnoses[0] if diagnoses else {}
    fus = h.get("follow_ups") or []
    project = h.get("project", {}) or {}
    age_days = d0.get("age_at_diagnosis")
    # Write LUAD-standard columns
    out = {c: "" for c in LUAD_REFERENCE_COLUMNS}
    out["submitter_id"] = (h.get("submitter_id") or "").upper()
    out["case_id"] = h.get("id") or ""
    out["project_id"] = project.get("project_id", "")
    # Vital status cascade: demographic > first diagnosis > first follow_up > empty
    vs = (demo.get("vital_status") or "").title()
    if not vs:
        vs = (d0.get("vital_status") or "").title() if isinstance(d0, dict) else ""
    if not vs and fus:
        fu0 = fus[0] if isinstance(fus[0], dict) else {}
        vs = (fu0.get("vital_status") or "").title()
    out["vital_status"] = vs
    # days_to_death cascade
    dth = demo.get("days_to_death")
    if dth is None and isinstance(d0, dict):
        dth = d0.get("days_to_death")
    # days_to_last_follow_up cascade
    lfu = demo.get("days_to_last_follow_up")
    if lfu is None and isinstance(d0, dict):
        lfu = d0.get("days_to_last_follow_up")
    if lfu is None and fus:
        for fu in fus:
            if isinstance(fu, dict):
                fl = fu.get("days_to_last_follow_up")
                if fl is not None:
                    if lfu is None or (str(fl) != "'--" and str(lfu) == "'--"):
                        lfu = fl
    # Clean '-- sentinel'
    def _clean_int(x):
        if x is None:
            return ""
        s = str(x).strip()
        if s in ("", "'--", "--", "NA", "N/A", "Not Reported", "Not Applicable", "Not Available", "Unknown", "[Not Available]", "[Not Applicable]", "[Discrepancy]", "[Completed]"):
            return ""
        try:
            v = int(float(s))
            return v if v >= 0 else ""
        except Exception:
            return ""
    dd_clean = _clean_int(dth)
    fl_clean = _clean_int(lfu)
    out["days_to_death"] = dd_clean
    out["days_to_last_follow_up"] = fl_clean
    out["n_diagnoses"] = len(diagnoses) if diagnoses else 0
    out["ajcc_pathologic_stage"] = d0.get("ajcc_pathologic_stage") or "" if isinstance(d0, dict) else ""
    out["ajcc_pathologic_t"] = d0.get("ajcc_pathologic_t") or "" if isinstance(d0, dict) else ""
    out["tumor_stage"] = d0.get("tumor_stage") or "" if isinstance(d0, dict) else ""
    out["grade"] = d0.get("tumor_grade") or "" if isinstance(d0, dict) else ""
    out["age_at_initial_pathologic_diagnosis"] = demo.get("age_at_initial_pathologic_diagnosis") or ""
    out["sex"] = demo.get("gender") or ""
    out["race"] = demo.get("race") or ""
    out["ajcc_pathologic_tumor_stage"] = d0.get("ajcc_pathologic_stage") or "" if isinstance(d0, dict) else ""
    out["histological_grade"] = d0.get("tumor_grade") or "" if isinstance(d0, dict) else ""
    out["initial_pathologic_dx_year"] = d0.get("year_of_diagnosis") or "" if isinstance(d0, dict) else ""
    out["histological_type"] = d0.get("primary_diagnosis") or "" if isinstance(d0, dict) else ""
    # Map to os/dss: fill both from vital_status (same as LUAD clinical convention)
    vs = out["vital_status"]
    dd = out["days_to_death"]
    fl = out["days_to_last_follow_up"]
    event_time = ""
    censorship = ""
    if vs == "Dead" and dd != "" and int(dd) >= 0:
        event_time = int(dd)
        censorship = 0.0
    elif vs == "Alive" and fl != "" and int(fl) >= 0:
        event_time = int(fl)
        censorship = 1.0
    if event_time != "":
        out["os_censorship"] = censorship
        out["os_survival_days"] = event_time
        out["dss_censorship"] = censorship
        out["dss_survival_days"] = event_time
        out["last_contact_days_to"] = event_time if censorship == 1.0 else ""
        out["death_days_to"] = event_time if censorship == 0.0 else ""
    return out


def main():
    global END
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdc-api-base", default=os.environ.get("GDC_API_BASE", END))
    ap.add_argument("--cohorts", nargs="+", required=True,
                    choices=["BRCA", "LUAD", "LUSC", "BLCA", "COAD", "READ", "PRAD", "KIRC", "HNSC", "LGG", "PAAD", "SKCM", "TGCT", "THCA", "UCEC"])
    ap.add_argument("--dry-run", action="store_true", help="print counts only, no write")
    args = ap.parse_args()
    END = str(args.gdc_api_base).rstrip("/")

    print(f"=== GDC /cases pull for cohorts: {args.cohorts} ===")
    summary = []
    for cohort in args.cohorts:
        project_id = f"TCGA-{cohort}"
        print(f"\n--- {project_id} ---")
        try:
            rows = pull_cohort_cases(project_id)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            continue
        df = pd.DataFrame(rows, columns=LUAD_REFERENCE_COLUMNS)
        # Stats
        n = len(df)
        n_dead = int(((df["vital_status"] == "Dead") & (df["days_to_death"] != "")).sum())
        n_alive_fu = int(((df["vital_status"] == "Alive") & (df["days_to_last_follow_up"] != "")).sum())
        n_usable = n_dead + n_alive_fu
        n_w_os = int(df["os_survival_days"].apply(lambda x: x != "" and x is not None).sum())
        print(f"  GDC total cases: {n}")
        print(f"  Dead+days_to_death: {n_dead}")
        print(f"  Alive+follow_up: {n_alive_fu}")
        print(f"  Total usable (OS/DSS complete): {n_usable} ({100*n_usable/max(n,1):.1f}%)")
        print(f"  Rows with os_survival_days filled: {n_w_os}")
        vs_dist = Counter(r["vital_status"] for r in rows)
        print(f"  Vital status dist: {dict(vs_dist)}")

        if args.dry_run:
            print("  [dry-run] skipped write")
            summary.append({"cohort": cohort, "gdc_total": n, "usable": n_usable, "written": False})
            continue

        # Write clinical.csv (LUAD-standard 60+ cols order)
        out_dir = OUT_CLIN_DIR / cohort
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / "clinical.csv"
        # Backup old if exists
        if out_p.exists():
            bak = out_dir / f"clinical.csv.bak_before_GDC_{int(time.time())}"
            out_p.rename(bak)
            print(f"  Backed up old clinical.csv → {bak.name}")
        df.to_csv(out_p, index=False)
        print(f"  ✅ Written → {out_p} ({out_p.stat().st_size/1024:.1f} KB)")
        summary.append({"cohort": cohort, "gdc_total": n, "usable": n_usable, "written": True})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"  {s['cohort']}: GDC={s['gdc_total']}, Usable={s['usable']}, Written={s['written']}")


if __name__ == "__main__":
    main()
