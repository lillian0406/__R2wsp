#!/usr/bin/env python3
"""BLCA 450 slides 412 cases 下载（requests.stream 永久修 6 坑）：
1. 读 J3_BLCA_DOWNLOAD_CAPPED450_FILEIDS_ONLY.txt（每行 file_id）
2. 读 J3_BLCA_DOWNLOAD_CAPPED450_412CASES.csv（拿 file_name/md5/size）
3. 每个 file_id：requests.get(stream=True, timeout=900) + 5 次指数退避 retry
4. .part 断点续传（支持 Range 断点）+ md5+size 双校验
5. 下完一张立刻 ln -s 软链到 data/raw_svs（GPU 边下边提并行不浪费）
"""
import argparse, csv, ctypes, hashlib, json, os, re, sys, threading, time
from pathlib import Path
import requests
import socket

socket.setdefaulttimeout(120)

PROJ = Path(__file__).resolve().parent.parent
END = os.environ.get("GDC_API_BASE", "https://api.gdc.cancer.gov").rstrip("/")
DATA_END = os.environ.get("GDC_DATA_BASE", f"{END}/data").rstrip("/")


class _ChunkIdleWatchdog:
    """Weak net hard-fix #7: requests.iter_content chunk idle watchdog.

    requests timeout=(connect, read) only controls first-byte / per-recv()
    syscall max time. In trans-pacific weak links it's VERY common that
    HTTP 200 headers come back in 5s, then iter_content() yields ZERO bytes
    for 3+ minutes (TCP zero window / middlebox stall). The thread below
    tracks wall-clock between successive non-empty chunks; if it exceeds
    `idle_seconds` we forcibly raise TimeoutError on the main downloader
    thread via PyThreadState_SetAsyncExc(TimeoutError).

    Typical symptom cured: HTTP 200 fast -> 0 bytes for 200s -> downloader
    stuck forever because no socket error is ever raised (TCP is alive but
    the window is 0, or a transparent proxy buffers-and-stalls).

    Exact GDC BLCA evidence 2026-08-03 18:47-19:00 China -> GDC Maryland:
    - curl -sI -> 400 OK 0.3s (DNS+TCP works)
    - curl -o... data endpoint -> 10s+ HTTP=000 size=0B speed=0
    - python requests.get(stream=True) 200 headers -> iter_content 0 chunk
      for 120s then process still alive (%cpu=0.0) = exactly this stall.
    """

    def __init__(self, idle_seconds: float, label: str = ""):
        self.idle = float(idle_seconds)
        self.label = label
        self._t = None
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._tid = None

    def kick(self) -> None:
        self._last = time.monotonic()

    def _runner(self) -> None:
        while not self._stop.is_set():
            if self._tid is not None and (time.monotonic() - self._last) > self.idle:
                try:
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(self._tid), ctypes.py_object(TimeoutError)
                    )
                    if res != 0:
                        print(
                            f"[watchdog {self.label}] idle>{self.idle:.0f}s -> "
                            f"injected TimeoutError into tid={self._tid}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return
                except Exception as e:  # noqa
                    print(f"[watchdog {self.label}] inject failed: {e}", file=sys.stderr, flush=True)
                    return
            time.sleep(max(1.0, min(5.0, self.idle * 0.1)))

    def __enter__(self):
        self._tid = threading.get_ident()
        self._t = threading.Thread(target=self._runner, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5.0)
        self._tid = None



def _resolve_fileid_info(fid: str, timeout: int) -> dict | None:
    """GDC files API 查单 file_id 的 file_name/md5/file_size/case_id/sample_barcode"""
    try:
        filters = {"op": "in", "content": {"field": "files.file_id", "value": [fid]}}
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,md5sum,cases.submitter_id,cases.samples.sample_barcode,cases.project.project_id",
            "format": "JSON",
            "size": "10",
        }
        r = requests.post(f"{END}/files", headers={"Content-Type": "application/json"}, json=params, timeout=timeout)
        r.raise_for_status()
        hits = r.json().get("data", {}).get("hits", [])
        if not hits:
            return None
        h = hits[0]
        cases = h.get("cases") or []
        sub = cases[0].get("submitter_id", "") if cases else ""
        samples = cases[0].get("samples", []) if cases else []
        bc = samples[0].get("sample_barcode", "") if samples else ""
        return dict(
            file_id=h["id"],
            file_name=h.get("file_name", f"{fid}.svs"),
            file_size=int(h.get("file_size") or 0),
            md5=h.get("md5sum", ""),
            case_id=sub,
            sample_barcode=bc,
            project_id=(cases[0].get("project", {}).get("project_id", "") if cases else ""),
        )
    except Exception as e:  # noqa
        print(f"[resolve] WARN {fid}: {e}", file=sys.stderr, flush=True)
        return None


def _resolve_batch(file_ids: list[str], timeout: int) -> dict[str, dict]:
    """批量 ≤100 POST body 查（永远不会 URL 414 过长）"""
    out: dict[str, dict] = {}
    for i in range(0, len(file_ids), 100):
        batch = file_ids[i : i + 100]
        filters = {"op": "in", "content": {"field": "files.file_id", "value": batch}}
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,md5sum,cases.submitter_id,cases.samples.sample_barcode,cases.project.project_id",
            "format": "JSON",
            "size": str(len(batch) * 4),
        }
        for attempt in range(1, 6):
            try:
                r = requests.post(
                    f"{END}/files",
                    headers={"Content-Type": "application/json"},
                    json=params,
                    timeout=timeout,
                )
                r.raise_for_status()
                hits = r.json().get("data", {}).get("hits", [])
                for h in hits:
                    cases = h.get("cases") or []
                    sub = cases[0].get("submitter_id", "") if cases else ""
                    samples = cases[0].get("samples", []) if cases else []
                    bc = samples[0].get("sample_barcode", "") if samples else ""
                    out[h["id"]] = dict(
                        file_id=h["id"],
                        file_name=h.get("file_name", f"{h['id']}.svs"),
                        file_size=int(h.get("file_size") or 0),
                        md5=h.get("md5sum", ""),
                        case_id=sub,
                        sample_barcode=bc,
                    )
                break
            except Exception as e:  # noqa
                wait = 2**attempt
                print(f"[resolve] batch {i//100+1} attempt {attempt}/5 failed: {e}; sleep {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
    return out


def _stream_download_one(
    info: dict,
    out_dir: Path,
    raw_svs_symlink: Path,
    timeout: int = 900,
    max_retries: int = 5,
    chunk_idle_seconds: int = 120,
) -> tuple[bool, str, str]:
    fid = info["file_id"]
    expected_size = int(info.get("file_size") or 0)
    expected_md5 = str(info.get("md5") or "").strip().lower()
    fn = info.get("file_name") or f"{fid}.svs"
    dest_dir = out_dir / fid
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / fn
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    symlink_target = raw_svs_symlink / fn

    if final_path.exists():
        sz = final_path.stat().st_size
        if expected_size and sz == expected_size:
            if final_path != symlink_target and not symlink_target.exists():
                try:
                    symlink_target.symlink_to(final_path.resolve())
                except Exception:
                    pass
            return True, str(final_path), f"exists-size-ok({sz}B)"
        else:
            print(f"[warn] {fn} exists size {sz}!={expected_size}, redownload", flush=True)
            final_path.unlink(missing_ok=True)
            part_path.unlink(missing_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Accept": "application/octet-stream"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            url = f"{DATA_END}/{fid}"
            with _ChunkIdleWatchdog(idle_seconds=chunk_idle_seconds, label=fn[:40]) as wd:
                with requests.get(url, stream=True, timeout=(60, timeout), headers=headers) as r:
                    if resume_from > 0 and r.status_code != 206:
                        print(f"[resume] server ignore Range (code={r.status_code}), restart from 0", flush=True)
                        resume_from = 0
                        part_path.unlink(missing_ok=True)
                    elif r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    r.raise_for_status()
                    mode = "ab" if resume_from > 0 else "wb"
                    h = hashlib.md5()
                    if resume_from > 0 and part_path.exists():
                        with part_path.open("rb") as rf:
                            for chunk in iter(lambda: rf.read(8 * 1024 * 1024), b""):
                                h.update(chunk)
                                wd.kick()
                    tot_bytes = resume_from
                    last_log = time.time()
                    wd.kick()
                    with part_path.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            h.update(chunk)
                            tot_bytes += len(chunk)
                            wd.kick()
                            now = time.time()
                            if now - last_log > 30:
                                print(
                                    f"  [{fn}] {tot_bytes/1024/1024:.1f}/{expected_size/1024/1024:.1f} MiB ({100*tot_bytes/max(1,expected_size):.1f}%)",
                                    flush=True,
                                )
                                last_log = now
            if expected_size and tot_bytes != expected_size:
                size_delta_pp = abs(tot_bytes - expected_size) / max(1, expected_size) * 100.0
                size_tolerate_pp = 0.1
                if size_delta_pp > size_tolerate_pp:
                    raise RuntimeError(f"size mismatch: got {tot_bytes} expect {expected_size} (Δ={size_delta_pp:.3f}% > {size_tolerate_pp}%)")
                else:
                    print(f"  [warn-size-tolerate] {fn}: Δ={size_delta_pp:.4f}% ≤ 0.1% → accept (got {tot_bytes} expect {expected_size})", flush=True)
            if expected_md5:
                got = h.hexdigest().lower()
                if got != expected_md5:
                    raise RuntimeError(f"md5 mismatch: got {got} expect {expected_md5}")
                md5_note = f"md5={got}"
            else:
                md5_note = f"size={tot_bytes}"
            part_path.replace(final_path)
            if not symlink_target.exists():
                try:
                    symlink_target.symlink_to(final_path.resolve())
                except Exception:
                    pass
            return True, str(final_path), md5_note
        except Exception as e:  # noqa
            wait = 2**attempt
            print(
                f"[download] {fn} attempt {attempt}/{max_retries} failed: {e}; sleep {wait}s",
                file=sys.stderr,
                flush=True,
            )
            if attempt == max_retries:
                return False, "", f"failed_after_{max_retries}_retries: {e}"
            time.sleep(wait)
    return False, "", "unexpected_state"


def main() -> None:
    global END, DATA_END
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdc-api-base", default=os.environ.get("GDC_API_BASE", END))
    ap.add_argument("--gdc-data-base", default=os.environ.get("GDC_DATA_BASE", DATA_END))
    ap.add_argument("--file-ids-txt", default=str(PROJ / "outputs/_disk_cohort_inventory/J3_BLCA_DOWNLOAD_CAPPED450_FILEIDS_ONLY.txt"))
    ap.add_argument("--capped-csv", default=str(PROJ / "outputs/_disk_cohort_inventory/J3_BLCA_DOWNLOAD_CAPPED450_412CASES.csv"))
    ap.add_argument("--out-dir", default=str(PROJ / "data/downloads/BLCA_capped450"))
    ap.add_argument("--raw-svs-symlink", default=str(PROJ / "data/raw_svs"))
    ap.add_argument("--manifest-out", default=str(PROJ / "outputs/_disk_cohort_inventory/J3_BLCA_CAPPED450_DOWNLOAD_MANIFEST.csv"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--chunk-idle-seconds", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="0=no limit")
    ap.add_argument("--resolve-only", action="store_true", default=False)
    args = ap.parse_args()
    END = str(args.gdc_api_base).rstrip("/")
    DATA_END = str(args.gdc_data_base).rstrip("/") if str(args.gdc_data_base).strip() else f"{END}/data"

    file_ids_txt = Path(args.file_ids_txt)
    capped_csv = Path(args.capped_csv)
    out_dir = Path(args.out_dir)
    raw_svs_symlink = Path(args.raw_svs_symlink)
    manifest_out = Path(args.manifest_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_svs_symlink.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    file_ids = [l.strip() for l in file_ids_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        file_ids = file_ids[: args.limit]
    print(f"[init] file_ids = {len(file_ids)} from {file_ids_txt}", flush=True)

    csv_info: dict[str, dict] = {}
    if capped_csv.exists():
        with capped_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fid = (row.get("file_id") or "").strip()
                if fid:
                    csv_info[fid] = dict(
                        file_id=fid,
                        file_name=(row.get("file_name") or f"{fid}.svs").strip(),
                        file_size=int(float(row.get("size_mb") or 0) * 1024 * 1024),
                        md5=(row.get("md5") or "").strip(),
                        case_id=(row.get("case_id_submitter") or "").strip(),
                        sample_barcode=(row.get("slide_barcode") or "").strip(),
                        priority_rank=(row.get("priority_rank") or "").strip(),
                        keep_reason=(row.get("keep_reason") or "").strip(),
                        sample_type=(row.get("sample_type") or "").strip(),
                    )
        print(f"[init] capped_csv info = {len(csv_info)} rows", flush=True)

    print("[resolve] GDC files API batch POST ≤100 (永久无 414 nginx URL 过长)...", flush=True)
    resolved = _resolve_batch(file_ids, args.timeout)
    print(f"[resolve] done resolved={len(resolved)}/{len(file_ids)}", flush=True)

    merged: list[dict] = []
    for fid in file_ids:
        info = dict(csv_info.get(fid, {}))
        r = resolved.get(fid)
        if r:
            for k, v in r.items():
                info.setdefault(k, v)
        info.setdefault("file_id", fid)
        info.setdefault("file_name", f"{fid}.svs")
        merged.append(info)

    fieldnames = [
        "file_id", "case_id", "sample_barcode", "sample_type", "priority_rank", "keep_reason",
        "file_name", "file_size", "md5", "download_status", "result_note", "local_path",
    ]
    rows_existing: dict[str, dict] = {}
    if manifest_out.exists():
        try:
            with manifest_out.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    fid = (row.get("file_id") or "").strip()
                    if fid:
                        rows_existing[fid] = row
        except Exception:
            pass

    def _write_rows(rows: list[dict]) -> None:
        with manifest_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    ok_n = 0
    fail_n = 0
    all_rows: list[dict] = list(rows_existing.values())
    existing_fids = set(rows_existing.keys())

    for idx, info in enumerate(merged):
        fid = info["file_id"]
        out_row = {k: info.get(k, "") for k in fieldnames}
        out_row["download_status"] = ""
        out_row["result_note"] = ""
        out_row["local_path"] = ""
        if fid in existing_fids and rows_existing[fid].get("download_status") == "OK":
            all_rows.append(rows_existing[fid])
            ok_n += 1
            continue
        if args.resolve_only:
            out_row["download_status"] = "RESOLVED"
            out_row["result_note"] = (
                f"file_size={info.get('file_size')} md5={info.get('md5')[:8]} case={info.get('case_id')}"
            )
            all_rows.append(out_row)
            continue
        try:
            ok, lp, note = _stream_download_one(
                info,
                out_dir=out_dir,
                raw_svs_symlink=raw_svs_symlink,
                timeout=args.timeout,
                chunk_idle_seconds=args.chunk_idle_seconds,
            )
        except Exception as e:  # noqa
            ok, lp, note = False, "", f"fatal: {e}"
        out_row["download_status"] = "OK" if ok else "FAIL"
        out_row["result_note"] = note
        out_row["local_path"] = lp
        all_rows.append(out_row)
        if ok:
            ok_n += 1
        else:
            fail_n += 1
        print(
            f"[{idx+1}/{len(merged)}] {out_row['download_status']} {info.get('case_id',''):<14} "
            f"{info.get('file_name','')[:70]:<70} "
            f"{int(info.get('file_size') or 0)/1024/1024:>7.1f} MiB  {note}",
            flush=True,
        )
        if (idx + 1) % 5 == 0 or idx == len(merged) - 1:
            _write_rows(all_rows)

    _write_rows(all_rows)
    total_cases = len({r.get("case_id") for r in all_rows if r.get("case_id")})
    total_size_mb = sum(int(r.get("file_size") or 0) for r in all_rows) / 1024 / 1024
    print("\n==================== SUMMARY ====================", flush=True)
    print(f"  total requested slides: {len(merged)}", flush=True)
    print(f"  OK = {ok_n}   FAIL = {fail_n}", flush=True)
    print(f"  unique cases covered  = {total_cases}", flush=True)
    print(f"  total manifest size   = {total_size_mb:.0f} MiB ({total_size_mb/1024:.1f} GiB)", flush=True)
    print(f"  manifest CSV          = {manifest_out}", flush=True)
    print(f"  downloads dir         = {out_dir}", flush=True)
    print(f"  raw_svs symlinks      = {raw_svs_symlink}", flush=True)
    print("=================================================", flush=True)


if __name__ == "__main__":
    main()
