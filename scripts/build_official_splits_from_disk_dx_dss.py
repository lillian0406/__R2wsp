from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from build_censored_stage_splits import stratified_kfold_case_ids, write_csv


@dataclass(frozen=True)
class CohortUniverse:
    cohort: str
    target_col: str
    cases: pd.DataFrame
    n_clinical_r1: int
    n_dx_svs: int
    n_dx_uni: int


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _case12(x: str) -> str:
    s = str(x).strip().upper()
    return s[:12] if s.startswith("TCGA-") and len(s) >= 12 else ""


def _load_clinical_r1_dss(cohort: str, *, target_col: str) -> tuple[pd.DataFrame, set[str]]:
    root = _root()
    p = root / "data" / "raw_clinical" / cohort / "clinical.csv"
    df = pd.read_csv(p)
    df["submitter_id"] = df["submitter_id"].astype(str).str.upper()
    df["case_id"] = df["submitter_id"].map(_case12)
    vs = df["vital_status"].astype(str).str.upper()
    ok_vs = vs.isin(["ALIVE", "DEAD"])
    censor_col = target_col.split("_")[0] + "_censorship"
    t = pd.to_numeric(df.get(target_col), errors="coerce")
    c = pd.to_numeric(df.get(censor_col), errors="coerce")
    ok = ok_vs & t.notna() & (t >= 0) & c.notna()
    out = df.loc[ok, ["case_id", "submitter_id", target_col, censor_col]].copy()
    out = out[out["case_id"] != ""].copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype(float)
    out[censor_col] = pd.to_numeric(out[censor_col], errors="coerce").astype(float)
    case_set = set(out["case_id"].astype(str).tolist())
    return out, case_set


def _scan_dx_svs_cases(*, cohort_case_set: set[str]) -> set[str]:
    root = _root()
    raw_svs = root / "data" / "raw_svs"
    out: set[str] = set()
    for p in raw_svs.rglob("*.svs"):
        stem = p.stem.upper()
        if not stem.startswith("TCGA-") or "-DX" not in stem:
            continue
        c = stem[:12]
        if c in cohort_case_set:
            out.add(c)
    return out


def _scan_dx_uni_cases_and_slide_map(*, cohort_case_set: set[str]) -> tuple[set[str], dict[str, str]]:
    root = _root()
    h5_dir = root / "data" / "wsi_features" / "extracted_mag20x_patch256_fp" / "uni1024" / "feats_h5"
    cases: set[str] = set()
    slide_by_case: dict[str, str] = {}
    for p in sorted(h5_dir.glob("*.h5")):
        stem = p.stem.upper()
        if not stem.startswith("TCGA-") or "-DX" not in stem:
            continue
        c = stem[:12]
        if c not in cohort_case_set:
            continue
        cases.add(c)
        slide_by_case.setdefault(c, p.stem)
    return cases, slide_by_case


def _load_case_whitelist(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = str(line).strip().upper()
        if not s:
            continue
        c = _case12(s)
        if c:
            out.add(c)
    return out


def _scan_rna_cases_from_tsv(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    header = tsv_path.open("r", encoding="utf-8", errors="replace").readline().strip("\n")
    cols = header.split("\t")
    cases: set[str] = set()
    for x in cols[1:]:
        s = str(x).strip().upper()
        c = _case12(s)
        if c:
            cases.add(c)
    return cases


def build_universe(
    cohort: str,
    *,
    target_col: str,
    require_dx_svs: bool,
    require_dx_uni: bool,
    require_rna: bool,
    rna_tsv_path: str | None,
    case_whitelist_file: str | None,
) -> CohortUniverse:
    clin_df, clin_cases = _load_clinical_r1_dss(cohort, target_col=target_col)
    dx_svs = _scan_dx_svs_cases(cohort_case_set=clin_cases) if require_dx_svs else set(clin_cases)
    dx_uni, slide_by_case = _scan_dx_uni_cases_and_slide_map(cohort_case_set=clin_cases) if require_dx_uni else (set(clin_cases), {})
    wl = _load_case_whitelist(case_whitelist_file)
    rna_cases = set(clin_cases)
    if require_rna:
        if rna_tsv_path:
            rna_cases = _scan_rna_cases_from_tsv(Path(rna_tsv_path))
        else:
            auto = _root() / "data" / "raw_rna" / "tpm_tsv" / f"{cohort.lower()}_tpm.tsv"
            rna_cases = _scan_rna_cases_from_tsv(auto)

    universe_cases = clin_cases
    if require_dx_svs:
        universe_cases = universe_cases & dx_svs
    if require_dx_uni:
        universe_cases = universe_cases & dx_uni
    if require_rna:
        universe_cases = universe_cases & rna_cases
    if wl is not None:
        universe_cases = universe_cases & wl

    censor_col = target_col.split("_")[0] + "_censorship"
    rows = clin_df[clin_df["case_id"].isin(sorted(universe_cases))].copy()
    rows["slide_id"] = rows["case_id"].map(lambda c: slide_by_case.get(str(c), str(c)))
    rows["project_id"] = f"TCGA-{cohort}"
    rows = rows.rename(columns={"case_id": "case_id", "submitter_id": "submitter_id"})
    rows = rows[["case_id", "slide_id", "submitter_id", "project_id", target_col, censor_col]].copy()

    return CohortUniverse(
        cohort=cohort,
        target_col=target_col,
        cases=rows,
        n_clinical_r1=int(len(clin_cases)),
        n_dx_svs=int(len(dx_svs)),
        n_dx_uni=int(len(dx_uni)),
    )


def write_official_kfold_splits(
    uni: CohortUniverse,
    *,
    out_root: Path,
    n_folds: int,
    seed: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = uni.cases.to_dict(orient="records")
    fold_ids = stratified_kfold_case_ids(rows, target_col=str(uni.target_col), n_folds=int(n_folds), seed=int(seed))
    all_case_ids = set(str(r["case_id"]) for r in rows)
    seen_test: set[str] = set()

    for k in range(int(n_folds)):
        test_ids = set(fold_ids[k])
        train_ids = all_case_ids - test_ids
        if train_ids & test_ids:
            raise RuntimeError("train/test overlap")
        if seen_test & test_ids:
            raise RuntimeError("test folds overlap")
        seen_test |= test_ids

        k_dir = out_root / f"k={k}"
        write_csv(k_dir / "train.csv", [r for r in rows if str(r["case_id"]) in train_ids])
        write_csv(k_dir / "test.csv", [r for r in rows if str(r["case_id"]) in test_ids])

    if seen_test != all_case_ids:
        raise RuntimeError(f"test coverage mismatch: {len(seen_test)} vs {len(all_case_ids)}")

    manifest = {
        "cohort": str(uni.cohort),
        "target_col": str(uni.target_col),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "universe": {
            "clinical_r1_cases": int(uni.n_clinical_r1),
            "dx_svs_cases": int(uni.n_dx_svs),
            "dx_uni_cases": int(uni.n_dx_uni),
            "final_cases": int(len(all_case_ids)),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out_cohort", required=True)
    ap.add_argument("--target_col", default="dss_survival_days")
    ap.add_argument("--require_dx_svs", action="store_true")
    ap.add_argument("--require_dx_uni", action="store_true")
    ap.add_argument("--require_rna", action="store_true")
    ap.add_argument("--rna_tsv_path", default=None)
    ap.add_argument("--case_whitelist_file", default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cohort = str(args.cohort).strip().upper()
    out_cohort = str(args.out_cohort).strip()
    root = _root()
    out_root = (root / "data" / "splits" / out_cohort).resolve()

    uni = build_universe(
        cohort,
        target_col=str(args.target_col),
        require_dx_svs=bool(args.require_dx_svs),
        require_dx_uni=bool(args.require_dx_uni),
        require_rna=bool(args.require_rna),
        rna_tsv_path=str(args.rna_tsv_path) if args.rna_tsv_path else None,
        case_whitelist_file=str(args.case_whitelist_file) if args.case_whitelist_file else None,
    )
    write_official_kfold_splits(uni, out_root=out_root, n_folds=int(args.n_folds), seed=int(args.seed))
    print(f"[OK] official splits written: {out_root}")


if __name__ == "__main__":
    main()
