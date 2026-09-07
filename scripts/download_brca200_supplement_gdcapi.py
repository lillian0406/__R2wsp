#!/usr/bin/env python3
"""
B2 · BRCA 补 200 病例 下载 SVS：用 GDC REST API (requests) 拿 UUID → HTTP 流式下载，下完 ln 到 raw_svs
(彻底不用 gdc-client 二进制，避免 unzip/url 错误)
输出目录: data/downloads/BRCA_supp200/<file_id>/<filename>
"""
import os, json, hashlib, time, argparse, csv
from pathlib import Path
import requests
PROJ = Path(__file__).resolve().parent.parent
END = os.environ.get("GDC_API_BASE", "https://api.gdc.cancer.gov").rstrip("/")
DATA_END = os.environ.get("GDC_DATA_BASE", f"{END}/data").rstrip("/")

def _infer_tcga_cohort_from_project_id(project_id: str) -> str | None:
    s = str(project_id or "").strip().upper()
    if s.startswith("TCGA-") and len(s) > 5:
        return s.split("-", 1)[1]
    return None


def _case12(x: str) -> str:
    s = str(x or "").strip().upper()
    return s[:12] if s.startswith("TCGA-") and len(s) >= 12 else ""


def _is_dx_file_name(file_name: str) -> bool:
    s = str(file_name or "").upper()
    return "-DX" in s


def _filter_cases_by_clinical_event(
    case_ids: list[str],
    *,
    cohort: str,
    clinical_csv: str | None = None,
    time_col: str = "dss_survival_days",
    censorship_col: str = "dss_censorship",
    case_col: str = "submitter_id",
) -> tuple[list[str], list[str]]:
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas-required-for-clinical-check") from e

    root = Path(__file__).resolve().parents[1]
    clinical_path = Path(clinical_csv) if clinical_csv else root / "data" / "raw_clinical" / cohort / "clinical.csv"
    df = pd.read_csv(clinical_path)
    if case_col not in df.columns:
        raise ValueError(f"case_col not in clinical.csv: {case_col}")
    if time_col not in df.columns:
        raise ValueError(f"time_col not in clinical.csv: {time_col}")
    if censorship_col not in df.columns:
        raise ValueError(f"censorship_col not in clinical.csv: {censorship_col}")

    df["_case12"] = df[case_col].astype(str).str.upper().str[:12]
    t = pd.to_numeric(df[time_col], errors="coerce")
    c = pd.to_numeric(df[censorship_col], errors="coerce")
    ok = t.notna() & (t >= 0) & c.notna() & c.isin([0, 1])
    event = ok & (c == 0)
    event_cases = set(df.loc[event, "_case12"].astype(str).tolist())

    kept = []
    dropped = []
    for cid in case_ids:
        c12 = _case12(cid)
        if c12 and c12 in event_cases:
            kept.append(c12)
        else:
            dropped.append(c12 or str(cid).strip().upper())
    kept = sorted(set(kept))
    dropped = sorted(set(dropped))
    return kept, dropped


def query_files_by_cases(
    case_ids,
    project_id: str,
    one_per_case: bool,
    prefer_dx: bool,
    *,
    require_dx: bool,
):
    """GDC files endpoint: 查项目 Slide Image (.svs) 对应 case_ids -> list[dict(file_id=uuid, file_name=.., file_size=.., md5=.., case_submitter_id=.., sample_barcode=..)]"""
    filters = {
        "op": "and",
        "content": [
            {"op":"in","content":{"field":"cases.submitter_id","value":[]}},
            {"op":"in","content":{"field":"cases.project.project_id","value":[project_id]}},
            {"op":"=","content":{"field":"data_type","value":"Slide Image"}},
            {"op":"in","content":{"field":"access","value":["open"]}},
        ]
    }
    # 分页一次 3000
    fields = "file_id,file_name,file_size,md5sum,cases.submitter_id,cases.samples.sample_barcode,analysis.input_files.file_size"
    allhits: list[dict] = []
    for chunk in [case_ids[i:i+100] for i in range(0,len(case_ids),100)]:
        filters["content"][0]["content"]["value"] = chunk
        params = {"filters": json.dumps(filters), "fields": fields, "format":"JSON", "size":3000}
        r = requests.post(f"{END}/files", headers={"Content-Type":"application/json"}, json=params, timeout=120)
        r.raise_for_status()
        hits = r.json().get('data',{}).get('hits',[])
        for h in hits:
            cases = h.get('cases',[]) or []
            sub = cases[0].get('submitter_id') if cases else ''
            samples = cases[0].get('samples',[]) if cases else []
            bc = samples[0].get('sample_barcode') if samples else ''
            allhits.append(dict(file_id=h['id'], file_name=h['file_name'],
                               file_size=int(h.get('file_size') or 0),
                               md5=h.get('md5sum',''), case_submitter_id=sub, sample_barcode=bc))
    if not one_per_case:
        return allhits
    by_case: dict[str, list[dict]] = {}
    for h in allhits:
        by_case.setdefault(h.get("case_submitter_id") or "", []).append(h)
    picked: list[dict] = []
    for case_id, hs in by_case.items():
        if not case_id:
            continue
        if prefer_dx:
            dx = [x for x in hs if _is_dx_file_name(x.get("file_name", ""))]
            if dx:
                picked.append(sorted(dx, key=lambda x: x.get("file_name", ""))[0])
                continue
            if require_dx:
                continue
        if require_dx:
            dx = [x for x in hs if _is_dx_file_name(x.get("file_name", ""))]
            if not dx:
                continue
            picked.append(sorted(dx, key=lambda x: x.get("file_name", ""))[0])
            continue
        picked.append(sorted(hs, key=lambda x: x.get("file_name", ""))[0])
    return picked

def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def download_one_with_resume(
    session: requests.Session,
    file_id: str,
    file_name: str,
    out_dir: str,
    expected_md5: str = "",
    expected_size: int = 0,
    *,
    max_retries: int = 8,
    connect_timeout: int = 30,
    read_timeout: int = 300,
    chunk_mb: int = 8,
):
    out_d = Path(out_dir) / file_id
    out_d.mkdir(parents=True, exist_ok=True)
    out_fp = out_d / Path(file_name).name
    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".partial")

    exp_size = int(expected_size or 0)
    exp_md5 = str(expected_md5 or "").strip().lower()

    if out_fp.exists() and exp_size > 0 and out_fp.stat().st_size == exp_size:
        return True, str(out_fp), "exists-size-ok"
    if exp_size > 0 and out_fp.exists() and out_fp.stat().st_size < exp_size and not tmp_fp.exists():
        out_fp.replace(tmp_fp)

    url = f"{DATA_END}/{file_id}"
    chunk_size = int(chunk_mb) * 1024 * 1024

    for attempt in range(int(max_retries)):
        try:
            existing = tmp_fp.stat().st_size if tmp_fp.exists() else 0

            headers = {"Accept": "application/octet-stream"}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            with session.get(
                url,
                stream=True,
                timeout=(int(connect_timeout), int(read_timeout)),
                headers=headers,
            ) as r:
                if existing > 0 and r.status_code == 200:
                    existing = 0
                    mode = "wb"
                    tmp_fp.unlink(missing_ok=True)
                    raise RuntimeError("range-not-supported-restart")

                r.raise_for_status()

                with open(tmp_fp, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)

            got = tmp_fp.stat().st_size if tmp_fp.exists() else 0
            if exp_size > 0 and got < exp_size:
                raise RuntimeError(f"incomplete-size got={got} expected={exp_size}")
            if exp_size > 0 and got > exp_size:
                raise RuntimeError(f"oversize got={got} expected={exp_size}")

            if exp_md5:
                md5 = _md5_file(tmp_fp)
                if md5.lower() != exp_md5:
                    raise RuntimeError(f"md5-mismatch got={md5} expected={exp_md5}")

            tmp_fp.replace(out_fp)
            return True, str(out_fp), "ok"
        except Exception as e:
            if attempt + 1 >= int(max_retries):
                return False, str(out_fp), str(e)
            time.sleep(min(120, 2 ** attempt))

def main():
    global END, DATA_END
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdc-api-base", default=os.environ.get("GDC_API_BASE", END))
    ap.add_argument("--gdc-data-base", default=os.environ.get("GDC_DATA_BASE", DATA_END))
    ap.add_argument("--project_id", default="TCGA-BRCA")
    ap.add_argument('--case-ids-file', required=True)
    ap.add_argument('--out-dir', default=str(PROJ/'data/downloads/BRCA_supp200'))
    ap.add_argument('--manifest-out', default=str(PROJ/'outputs/_disk_cohort_inventory/BRCA_supp200_GDC_manifest_CASE_API.csv'))
    ap.add_argument('--raw-svs-symlink', default=str(PROJ/'data/raw_svs'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--one-per-case', action='store_true')
    ap.add_argument('--prefer-dx', action='store_true')
    ap.add_argument('--require-dx', action='store_true')
    ap.add_argument('--clinical-check-event', action='store_true')
    ap.add_argument('--clinical-csv', default=None)
    ap.add_argument('--clinical-time-col', default='dss_survival_days')
    ap.add_argument('--clinical-censorship-col', default='dss_censorship')
    ap.add_argument('--clinical-case-col', default='submitter_id')
    ap.add_argument('--eligible-cases-out', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-retries', type=int, default=8)
    ap.add_argument('--connect-timeout', type=int, default=30)
    ap.add_argument('--read-timeout', type=int, default=300)
    ap.add_argument('--chunk-mb', type=int, default=8)
    args = ap.parse_args()
    END = str(args.gdc_api_base).rstrip("/")
    DATA_END = str(args.gdc_data_base).rstrip("/") if str(args.gdc_data_base).strip() else f"{END}/data"

    case_ids = [l.strip().upper() for l in open(args.case_ids_file) if l.strip().upper().startswith("TCGA-")]
    case_ids = sorted({_case12(x) for x in case_ids if _case12(x)})

    dropped_clinical: list[str] = []
    if bool(args.clinical_check_event):
        cohort = _infer_tcga_cohort_from_project_id(str(args.project_id)) or "UNKNOWN"
        kept, dropped = _filter_cases_by_clinical_event(
            case_ids,
            cohort=cohort,
            clinical_csv=args.clinical_csv,
            time_col=str(args.clinical_time_col),
            censorship_col=str(args.clinical_censorship_col),
            case_col=str(args.clinical_case_col),
        )
        dropped_clinical = dropped
        case_ids = kept

    print(f"[API QUERY] project={args.project_id} cases={len(case_ids)} → GDC files open SVS")
    files = query_files_by_cases(
        case_ids,
        project_id=str(args.project_id),
        one_per_case=bool(args.one_per_case),
        prefer_dx=bool(args.prefer_dx),
        require_dx=bool(args.require_dx),
    )
    print(f"[QUERY DONE] {len(files)} files:")
    uniq_cases_in = sorted({f['case_submitter_id'] for f in files if f['case_submitter_id']})
    print(f"  unique case covered = {len(uniq_cases_in)}/{len(case_ids)} (missed = {sorted(set(case_ids)-set(uniq_cases_in))[:10]})")
    if bool(args.clinical_check_event):
        print(f"  clinical_event_dropped = {len(dropped_clinical)}")

    eligible_cases = sorted(set(uniq_cases_in))
    if str(args.eligible_cases_out).strip():
        outp = Path(args.eligible_cases_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(eligible_cases) + ("\n" if eligible_cases else ""), encoding="utf-8")
        print(f"[eligible] cases={len(eligible_cases)} path={outp}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.raw_svs_symlink).mkdir(parents=True, exist_ok=True)
    mf = Path(args.manifest_out); mf.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.dry_run):
        with open(mf, 'w', newline='') as mf_f:
            fw = csv.DictWriter(mf_f, fieldnames=['case_submitter_id','sample_barcode','file_id','file_name','file_size','md5','download_status','result_note','local_path'])
            fw.writeheader()
            for f in files:
                fw.writerow({**f,'download_status':'PLAN','result_note':'dry-run','local_path':''})
        print(f"\n✅ DRY-RUN DONE: planned={len(files)} files; manifest CSV: {mf}")
        return

    session = requests.Session()
    done_c = 0
    with open(mf, 'w', newline='') as mf_f:
        fw = csv.DictWriter(mf_f, fieldnames=['case_submitter_id','sample_barcode','file_id','file_name','file_size','md5','download_status','result_note','local_path'])
        fw.writeheader()
        mf_f.flush()
        for i, f in enumerate(files):
            if args.limit and i >= args.limit:
                break
            ok, local, note = download_one_with_resume(
                session,
                f['file_id'],
                f['file_name'],
                args.out_dir,
                expected_md5=f['md5'],
                expected_size=f['file_size'],
                max_retries=int(args.max_retries),
                connect_timeout=int(args.connect_timeout),
                read_timeout=int(args.read_timeout),
                chunk_mb=int(args.chunk_mb),
            )
            lp = Path(local)
            if ok and lp.exists():
                target = Path(args.raw_svs_symlink) / lp.name
                if not target.exists():
                    try:
                        target.symlink_to(lp.resolve())
                    except Exception:
                        pass
            st = 'OK' if ok else 'ERR'
            print(f"[{i+1}/{len(files)}] {st} {f['case_submitter_id']:<16} {f['file_name']:<80} {f['file_size']/1024/1024:>7.1f} MiB  {note}")
            fw.writerow({**f,'download_status':st,'result_note':str(note)[:200],'local_path':str(local) if ok else ''})
            mf_f.flush()
            if ok:
                done_c += 1
    print(f"\n✅ DONE: {done_c}/{len(files)} files; manifest CSV: {mf}")

if __name__ == '__main__': main()
