#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


PROJ = Path(__file__).resolve().parent.parent
END = os.environ.get("GDC_API_BASE", "https://api.gdc.cancer.gov").rstrip("/")
DATA_END = os.environ.get("GDC_DATA_BASE", f"{END}/data").rstrip("/")


@dataclass(frozen=True)
class GDCFileHit:
    file_id: str
    file_name: str
    file_size: int
    md5: str
    case_id: str
    sample_id: str
    aliquot_id: str

    @property
    def column_name(self) -> str:
        return self.aliquot_id or self.sample_id or self.case_id


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _query_star_counts_files(*, project_id: str, size: int = 20000) -> list[GDCFileHit]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = ",".join(
        [
            "file_id",
            "file_name",
            "file_size",
            "md5sum",
            "cases.submitter_id",
            "cases.samples.submitter_id",
            "cases.samples.sample_type",
            "cases.samples.portions.analytes.aliquots.submitter_id",
        ]
    )
    params = {"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": int(size)}
    r = requests.post(f"{END}/files", headers={"Content-Type": "application/json"}, json=params, timeout=120)
    r.raise_for_status()
    hits = r.json().get("data", {}).get("hits", [])
    out: list[GDCFileHit] = []
    for h in hits:
        cases = h.get("cases", []) or []
        if not cases:
            continue
        case = cases[0] or {}
        case_id = str(case.get("submitter_id") or "").upper()
        samples = case.get("samples", []) or []
        sample_id = ""
        aliquot_id = ""
        if samples:
            sample = samples[0] or {}
            sample_id = str(sample.get("submitter_id") or "").upper()
            portions = sample.get("portions", []) or []
            if portions:
                analytes = (portions[0] or {}).get("analytes", []) or []
                if analytes:
                    aliquots = (analytes[0] or {}).get("aliquots", []) or []
                    if aliquots:
                        aliquot_id = str((aliquots[0] or {}).get("submitter_id") or "").upper()
        out.append(
            GDCFileHit(
                file_id=str(h.get("id") or ""),
                file_name=str(h.get("file_name") or ""),
                file_size=int(h.get("file_size") or 0),
                md5=str(h.get("md5sum") or ""),
                case_id=case_id,
                sample_id=sample_id,
                aliquot_id=aliquot_id,
            )
        )
    out = [x for x in out if x.file_id and x.column_name]
    out.sort(key=lambda x: (x.case_id, x.column_name, x.file_id))
    return out


def _download_one_with_resume(
    session: requests.Session,
    file_id: str,
    *,
    out_fp: Path,
    expected_md5: str = "",
    expected_size: int = 0,
    max_retries: int = 8,
    connect_timeout: int = 30,
    read_timeout: int = 300,
    chunk_mb: int = 8,
) -> tuple[bool, str]:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".partial")

    exp_size = int(expected_size or 0)
    exp_md5 = str(expected_md5 or "").strip().lower()

    if out_fp.exists() and exp_size > 0 and out_fp.stat().st_size == exp_size:
        if exp_md5:
            md5 = _md5_file(out_fp)
            if md5.lower() != exp_md5:
                out_fp.unlink(missing_ok=True)
            else:
                return True, "exists-size-md5-ok"
        else:
            return True, "exists-size-ok"

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
            return True, "ok"
        except Exception as e:
            if attempt + 1 >= int(max_retries):
                return False, str(e)
            time.sleep(min(120, 2**attempt))
    return False, "unreachable"


def _parse_star_counts_tpm(path: Path) -> tuple[list[str], np.ndarray]:
    genes: list[str] = []
    vals: list[float] = []
    tpm_idx = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if tpm_idx is None:
                if "gene_id" not in parts:
                    raise ValueError("unexpected STAR-Counts header")
                try:
                    tpm_idx = parts.index("tpm_unstranded")
                except ValueError as e:
                    raise ValueError("tpm_unstranded not found in header") from e
                continue
            gene_id = parts[0]
            if not gene_id.startswith("ENSG"):
                continue
            try:
                v = float(parts[tpm_idx])
            except Exception:
                v = 0.0
            genes.append(gene_id)
            vals.append(v)
    if not genes:
        raise ValueError(f"no ENSG genes parsed: {path}")
    return genes, np.asarray(vals, dtype=np.float32)


def _write_matrix_tsv(out_path: Path, genes: list[str], col_names: list[str], mat: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mat.shape != (len(genes), len(col_names)):
        raise ValueError(f"matrix shape mismatch: {mat.shape} vs {(len(genes), len(col_names))}")
    import pandas as pd

    df = pd.DataFrame(mat, index=genes, columns=col_names)
    df.to_csv(out_path, sep="\t")


def main() -> None:
    global END, DATA_END
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--project-id", default="")
    ap.add_argument("--out-path", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--connect-timeout", type=int, default=30)
    ap.add_argument("--read-timeout", type=int, default=300)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--manifest-out", default="")
    args = ap.parse_args()

    cohort = str(args.cohort).upper()
    project_id = str(args.project_id).strip() or f"TCGA-{cohort}"
    out_path = Path(args.out_path).expanduser() if str(args.out_path).strip() else (PROJ / "data" / "raw_rna" / "tpm_tsv" / f"{cohort.lower()}_tpm.tsv")
    cache_dir = Path(args.cache_dir).expanduser() if str(args.cache_dir).strip() else (PROJ / "data" / "downloads" / "rna_star_counts" / cohort)
    manifest_out = Path(args.manifest_out).expanduser() if str(args.manifest_out).strip() else (PROJ / "outputs" / "_disk_cohort_inventory" / f"{cohort}_rna_star_counts_manifest.csv")

    print(f"[GDC] cohort={cohort} project_id={project_id}")
    hits = _query_star_counts_files(project_id=project_id, size=int(args.size))
    if int(args.max_files) > 0:
        hits = hits[: int(args.max_files)]
    print(f"[QUERY DONE] files={len(hits)}")
    if not hits:
        raise SystemExit(2)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    genes_ref: list[str] | None = None
    col_names: list[str] = []
    vecs: list[np.ndarray] = []

    with open(manifest_out, "w", newline="") as mf:
        fw = csv.DictWriter(
            mf,
            fieldnames=[
                "case_id",
                "sample_id",
                "aliquot_id",
                "column_name",
                "file_id",
                "file_name",
                "file_size",
                "md5",
                "download_status",
                "parse_status",
                "note",
                "local_path",
            ],
        )
        fw.writeheader()
        mf.flush()

        for i, h in enumerate(hits):
            local_fp = cache_dir / f"{h.file_id}.tsv"
            ok, note = _download_one_with_resume(
                session,
                h.file_id,
                out_fp=local_fp,
                expected_md5=h.md5,
                expected_size=h.file_size,
                max_retries=int(args.max_retries),
                connect_timeout=int(args.connect_timeout),
                read_timeout=int(args.read_timeout),
                chunk_mb=int(args.chunk_mb),
            )
            d_st = "OK" if ok else "ERR"
            p_st = "SKIP"
            p_note = note
            if ok:
                try:
                    genes, v = _parse_star_counts_tpm(local_fp)
                    if genes_ref is None:
                        genes_ref = genes
                    elif genes != genes_ref:
                        raise ValueError("gene list mismatch")
                    col_names.append(h.column_name)
                    vecs.append(v)
                    p_st = "OK"
                    p_note = "ok"
                except Exception as e:
                    p_st = "ERR"
                    p_note = str(e)
            fw.writerow(
                dict(
                    case_id=h.case_id,
                    sample_id=h.sample_id,
                    aliquot_id=h.aliquot_id,
                    column_name=h.column_name,
                    file_id=h.file_id,
                    file_name=h.file_name,
                    file_size=h.file_size,
                    md5=h.md5,
                    download_status=d_st,
                    parse_status=p_st,
                    note=str(p_note)[:200],
                    local_path=str(local_fp) if ok else "",
                )
            )
            mf.flush()
            print(f"[{i+1}/{len(hits)}] DL={d_st} PARSE={p_st} {h.case_id:<16} {h.column_name:<30} {p_note}")

    if genes_ref is None or not vecs:
        raise SystemExit(3)
    mat = np.stack(vecs, axis=1)
    _write_matrix_tsv(out_path, genes_ref, col_names, mat)
    print(f"\n✅ DONE: {out_path}  shape=({len(genes_ref)}, {len(col_names)})  manifest={manifest_out}")


if __name__ == "__main__":
    main()

