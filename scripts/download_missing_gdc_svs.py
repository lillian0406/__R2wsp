from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


_GDC_API_BASE = os.environ.get("GDC_API_BASE", "https://api.gdc.cancer.gov").rstrip("/")
FILES_ENDPOINT = os.environ.get("GDC_FILES_ENDPOINT", f"{_GDC_API_BASE}/files").rstrip("/")
DATA_ENDPOINT = os.environ.get("GDC_DATA_BASE", f"{_GDC_API_BASE}/data").rstrip("/") + "/"


def _post_json(url: str, payload: dict, *, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_batch(file_names: list[str], *, timeout: int) -> list[dict]:
    filters = {
        "op": "in",
        "content": {"field": "file_name", "value": file_names},
    }
    payload = {
        "filters": filters,
        "format": "JSON",
        "size": str(max(100, len(file_names) * 4)),
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "file_size",
                "md5sum",
                "state",
                "cases.submitter_id",
                "cases.samples.sample_type",
                "cases.project.project_id",
            ]
        ),
    }
    obj = _post_json(FILES_ENDPOINT, payload, timeout=timeout)
    return obj.get("data", {}).get("hits", [])


def _download_file(
    *,
    file_id: str,
    file_name: str,
    out_dir: Path,
    timeout: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = DATA_ENDPOINT + urllib.parse.quote(file_id)
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def _read_slide_stems(txt_path: Path) -> list[str]:
    stems: list[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if not x:
            continue
        stems.append(x)
    return stems


def _write_manifest(rows: list[dict], manifest_csv: Path) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_slide_id",
        "requested_file_name",
        "matched",
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "state",
        "case_submitter_id",
        "sample_type",
        "project_id",
        "download_path",
        "error",
    ]
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_existing_manifest(manifest_csv: Path) -> dict[str, dict]:
    if not manifest_csv.exists():
        return {}
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(row.get("requested_slide_id", "")): row for row in reader if row.get("requested_slide_id")}


def _retry_resolve_batch(*, batch: list[str], timeout: int, retries: int, backoff_seconds: float) -> list[dict]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _query_batch(batch, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            print(
                f"[resolve] retry {attempt}/{retries} failed for batch size={len(batch)}: {e}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    if last_exc is not None:
        raise last_exc
    return []


def _retry_download(
    *,
    file_id: str,
    file_name: str,
    out_dir: Path,
    timeout: int,
    retries: int,
    backoff_seconds: float,
) -> Path:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _download_file(file_id=file_id, file_name=file_name, out_dir=out_dir, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            part_path = (out_dir / file_name).with_suffix(Path(file_name).suffix + ".part")
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            print(
                f"[download] retry {attempt}/{retries} failed for {file_name}: {e}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"unexpected retry state for {file_name}")


def main() -> None:
    global FILES_ENDPOINT, DATA_ENDPOINT
    p = argparse.ArgumentParser(description="Resolve and optionally download missing TCGA SVS files from GDC by slide file name.")
    p.add_argument("--gdc-api-base", default=os.environ.get("GDC_API_BASE", _GDC_API_BASE))
    p.add_argument("--gdc-data-base", default=os.environ.get("GDC_DATA_BASE", DATA_ENDPOINT.rstrip("/")))
    p.add_argument(
        "--input-txt",
        default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_missing_official_split_slides.txt",
        help="Text file containing one missing slide_id stem per line (without .svs).",
    )
    p.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/R2wsp/data/raw_svs",
        help="Directory to store downloaded .svs files.",
    )
    p.add_argument(
        "--manifest-csv",
        default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_missing_official_split_slides_resolved.csv",
        help="CSV manifest for resolved/downloaded files.",
    )
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--max-files", type=int, default=0, help="0 means no limit.")
    p.add_argument("--resolve-only", action="store_true", default=False)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--resolve-retries", type=int, default=5)
    p.add_argument("--download-retries", type=int, default=5)
    p.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    args = p.parse_args()
    gdc_api_base = str(args.gdc_api_base).rstrip("/")
    FILES_ENDPOINT = f"{gdc_api_base}/files"
    DATA_ENDPOINT = str(args.gdc_data_base).rstrip("/") + "/"

    input_txt = Path(args.input_txt).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()

    stems = _read_slide_stems(input_txt)
    if args.max_files > 0:
        stems = stems[: args.max_files]

    print("input_txt   =", input_txt, flush=True)
    print("output_dir  =", output_dir, flush=True)
    print("manifest_csv=", manifest_csv, flush=True)
    print("count       =", len(stems), flush=True)
    print("resolve_only=", args.resolve_only, flush=True)
    print("request_timeout =", args.request_timeout, flush=True)
    print("resolve_retries =", args.resolve_retries, flush=True)
    print("download_retries =", args.download_retries, flush=True)

    existing_rows = _read_existing_manifest(manifest_csv)
    if existing_rows:
        print("existing_manifest_rows =", len(existing_rows), flush=True)
    stems = [
        stem
        for stem in stems
        if not (
            (output_dir / f"{stem}.svs").exists()
            and (output_dir / f"{stem}.svs").stat().st_size > 0
        )
    ]
    print("remaining_after_existing_files =", len(stems), flush=True)

    queries = [f"{stem}.svs" for stem in stems]
    hits_by_name: dict[str, dict] = {}

    for i in range(0, len(queries), args.batch_size):
        batch = queries[i : i + args.batch_size]
        print(
            f"[resolve] batch {i // args.batch_size + 1} / {(len(queries) + args.batch_size - 1) // args.batch_size}, size={len(batch)}",
            flush=True,
        )
        try:
            hits = _retry_resolve_batch(
                batch=batch,
                timeout=args.request_timeout,
                retries=args.resolve_retries,
                backoff_seconds=args.retry_backoff_seconds,
            )
        except Exception as e:
            print(f"[resolve] batch failed: {e}", file=sys.stderr, flush=True)
            hits = []
        for hit in hits:
            name = str(hit.get("file_name", ""))
            if name and name not in hits_by_name:
                hits_by_name[name] = hit
        time.sleep(args.sleep_seconds)

    rows: list[dict] = list(existing_rows.values())
    downloaded = 0
    matched = 0
    unresolved = 0

    for stem in stems:
        file_name = f"{stem}.svs"
        hit = hits_by_name.get(file_name)
        row = {
            "requested_slide_id": stem,
            "requested_file_name": file_name,
            "matched": False,
            "file_id": "",
            "file_name": "",
            "file_size": "",
            "md5sum": "",
            "state": "",
            "case_submitter_id": "",
            "sample_type": "",
            "project_id": "",
            "download_path": "",
            "error": "",
        }
        if hit is None:
            unresolved += 1
            row["error"] = "not_found_in_gdc_api"
            rows.append(row)
            continue

        matched += 1
        row["matched"] = True
        row["file_id"] = str(hit.get("file_id", ""))
        row["file_name"] = str(hit.get("file_name", ""))
        row["file_size"] = str(hit.get("file_size", ""))
        row["md5sum"] = str(hit.get("md5sum", ""))
        row["state"] = str(hit.get("state", ""))

        cases = hit.get("cases") or []
        if cases:
            row["case_submitter_id"] = str(cases[0].get("submitter_id", ""))
            samples = cases[0].get("samples") or []
            if samples:
                row["sample_type"] = str(samples[0].get("sample_type", ""))
            project = cases[0].get("project") or {}
            row["project_id"] = str(project.get("project_id", ""))

        if not args.resolve_only:
            try:
                out_path = _retry_download(
                    file_id=row["file_id"],
                    file_name=row["file_name"],
                    out_dir=output_dir,
                    timeout=args.request_timeout,
                    retries=args.download_retries,
                    backoff_seconds=args.retry_backoff_seconds,
                )
                row["download_path"] = str(out_path)
                downloaded += 1
                print(f"[download] ok {downloaded}/{len(stems)} -> {out_path.name}", flush=True)
            except Exception as e:
                row["error"] = f"download_failed: {e}"
                print(f"[download] failed {file_name}: {e}", file=sys.stderr, flush=True)

        rows.append(row)
        _write_manifest(rows, manifest_csv)

    _write_manifest(rows, manifest_csv)
    print("matched    =", matched, flush=True)
    print("unresolved =", unresolved, flush=True)
    if not args.resolve_only:
        print("downloaded =", downloaded, flush=True)
    print("manifest   =", manifest_csv, flush=True)


if __name__ == "__main__":
    main()
