#!/usr/bin/env python3
import csv, os, sys, time, hashlib
from pathlib import Path
import urllib.request, ssl
ssl._create_default_https_context = ssl._create_unverified_context
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT/'outputs/_disk_cohort_inventory/J3_BLCA_GDC_SVS_MANIFEST_FINAL.csv'
OUT_DIR = ROOT/'data/downloads/BLCA_926_open_access'
RAW_SVS = ROOT/'data/raw_svs'
OUT_DIR.mkdir(parents=True, exist_ok=True); RAW_SVS.mkdir(parents=True, exist_ok=True)
DATA = "https://api.gdc.cancer.gov/data/"
TIMEOUT = 600; MAX_RETRY = 5

def md5_of(p: Path, chunk=16*1024*1024) -> str:
    h = hashlib.md5()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def download_one(file_id, file_name, expect_size=0):
    out_path = OUT_DIR/file_name
    # 已存在检查
    if out_path.exists() and out_path.stat().st_size > 0 and (expect_size==0 or out_path.stat().st_size==int(expect_size)):
        return 'EXIST', out_path
    part = out_path.with_suffix(out_path.suffix + '.part')
    if part.exists(): part.unlink()
    url = DATA + file_id
    for attempt in range(1, MAX_RETRY+1):
        try:
            req = urllib.request.Request(url, headers={"Accept":"application/octet-stream","User-Agent":"R2wsp-DL/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, part.open('wb') as fo:
                while True:
                    chunk = resp.read(8*1024*1024)
                    if not chunk: break
                    fo.write(chunk)
            if expect_size and part.stat().st_size != int(expect_size):
                print(f"  [WARN size_mismatch] {file_name} part_sz={part.stat().st_size} expect={expect_size} retry {attempt}", flush=True)
                part.unlink(); time.sleep(2**attempt); continue
            part.replace(out_path)
            return 'OK', out_path
        except Exception as e:
            print(f"  [ERR retry {attempt}/{MAX_RETRY}] {file_name}: {type(e).__name__}: {e}", flush=True)
            if part.exists():
                try: part.unlink()
                except: pass
            if attempt < MAX_RETRY: time.sleep(2**attempt)
    return 'FAIL', out_path

def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    todo = [(r['submitter_id'], r['file_id'], r['file_name'], int(float(r['size_mb'])*1024*1024 if r.get('size_mb') else 0)) for r in rows]
    print(f"[START] BLCA manifest rows = {len(todo)} (926 slides 412 cases)", flush=True)
    print(f"  out_dir  = {OUT_DIR}", flush=True)
    print(f"  raw_svs  = {RAW_SVS} (下完立刻软链)", flush=True)
    ok=0; ex=0; fail=0; total=len(todo); start=time.time(); bytes_dl=0
    for i,(sub,fid,fn,esz) in enumerate(todo,1):
        st, p = download_one(fid, fn, esz)
        if st=='EXIST': ex+=1
        elif st=='OK':
            ok+=1; bytes_dl += p.stat().st_size
            # 立刻软链到 raw_svs（用户说下完马上可以 UNI 提，GPU 不闲着）
            link = RAW_SVS/p.name
            if not link.exists():
                try: os.symlink(p.resolve(), link)
                except Exception as e: print(f"  [WARN symlink fail] {p.name}: {e}", flush=True)
        else: fail+=1
        if i%5==0 or i==total:
            eta = ''
            if ok>0 and (time.time()-start)>0:
                mbps = bytes_dl/1024/1024/max(0.1,time.time()-start)
                rem = (total-i)/max(0.001, ok/(time.time()-start))
                eta = f" speed={mbps:.1f}MB/s rem={int(rem//60)}m{int(rem%60)}s"
            print(f"  [{i}/{total}] OK={ok} EXIST={ex} FAIL={fail}{eta}  last={fn[:60]}", flush=True)
    print(f"[DONE] {ok+ex}/{total} success (OK={ok} EXIST={ex} FAIL={fail})", flush=True)
    # 最后检查软链到 raw_svs 的数量
    n_link = sum(1 for p in RAW_SVS.iterdir() if p.is_symlink() and 'BLCA' in str(p.resolve()).upper() or (lambda x: 'TCGA-' in x.name and x.resolve().parent.name.endswith('BLCA_926_open_access'))(p))
    print(f"[SYMLINK] BLCA raw_svs 软链新增 ≈ {ok} 个（软链路径：{RAW_SVS}/<file_name.svs>）", flush=True)

if __name__=='__main__': main()
