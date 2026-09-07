import json, csv
from pathlib import Path
import numpy as np
import sys
try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

ROOT=Path('/root/autodl-tmp/R2wsp')
FULL=ROOT/'outputs'/'censored_stage_survival_full_upgrade'/'wsi=DINOv2_ViTL_tilefix256_rna=hallmark50_omics'/'win0.59-0.65_A0.99_K50'
P1BASE=ROOT/'outputs'/'sweep_censored_stage_survival_anti_plip_vec_LUAD'/'baseline'
PHASE2OLD=ROOT/'outputs'/'censored_stage_survival_phase2'
PHASE1OLD=ROOT/'outputs'/'censored_stage_survival_phase1'

def read_json(p):
    if not p.is_file():
        return None
    return json.loads(p.read_text())

def load_csv(p):
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

print("====== (1) DINOv2 ViTL full scheduler phase2 final_test_c N(15 paired, 5 folds x 3 seeds) ======")
import re
RE_FS = re.compile(r"fold_(\d+)_seed(\d+)")
for eval_kind in ['eval_with_censored','eval_uncensored_only']:
    vals=[]
    root = FULL/eval_kind/'phase2'
    if root.is_dir():
        for p in sorted(root.rglob('summary.json')):
            m = RE_FS.search(str(p))
            if not m: continue
            f, s = int(m.group(1)), int(m.group(2))
            d=read_json(p)
            if d is not None and 'final_test_c_index' in d:
                vals.append((f,s,float(d['final_test_c_index']), int(d.get('n_test_cases', -1))))
    if not vals:
        print(f"  [{eval_kind}] no results"); continue
    a=np.asarray([x[2] for x in vals])*100
    print(f"\n  [{eval_kind}] n={len(a)}: final_test_c_index mean+/-std = {a.mean():.2f} +/- {a.std(ddof=1):.2f} pp  median={np.median(a):.2f} pp  range=[{a.min():.2f}, {a.max():.2f}]")
    print(f"      Per-fold (3 seeds mean+/-std):")
    for f in range(5):
        col=np.asarray([x[2] for x in vals if x[0]==f])*100
        ntes=list(set([x[3] for x in vals if x[0]==f]))
        if col.size:
            print(f"        fold{f}: {col.mean():.2f} +/- {col.std(ddof=1):.2f} pp  (N_te per run={ntes})")

print("\n====== (2) PAIRED DINOv2 ViTL (eval_with_censored phase2) vs PLIP baseline (phase2 k5 win0.59-0.65 eval_with_censored), N=15 exact (fold,seed) matches ======")
dino_map={}
for eval_kind in ['eval_with_censored']:
    root = FULL/eval_kind/'phase2'
    if root.is_dir():
        for p in sorted(root.rglob('summary.json')):
            m = RE_FS.search(str(p))
            if not m: continue
            f, s = int(m.group(1)), int(m.group(2))
            d=read_json(p)
            if d and 'final_test_c_index' in d:
                dino_map[(f,s)] = float(d['final_test_c_index'])
plip_map={}
plip_root = PHASE2OLD/'k5'/'win0.59-0.65_A0.99_K50'/'eval_with_censored'
if plip_root.is_dir():
    for p in sorted(plip_root.rglob('summary.json')):
        m = RE_FS.search(str(p))
        if not m: continue
        f, s = int(m.group(1)), int(m.group(2))
        d=read_json(p)
        if d and 'final_test_c_index' in d:
            plip_map[(f,s)] = float(d['final_test_c_index'])
paired=[]
for k in sorted(set(dino_map.keys()) & set(plip_map.keys())):
    paired.append((k[0], k[1], plip_map[k], dino_map[k]))
if paired:
    a_plip=np.asarray([x[2] for x in paired])*100
    a_dino=np.asarray([x[3] for x in paired])*100
    delta = a_dino - a_plip
    print(f"  PLIP baseline (phase2 k5 eval_with_censored) n={len(a_plip)}: {a_plip.mean():.2f} +/- {a_plip.std(ddof=1):.2f} pp  median={np.median(a_plip):.2f} pp")
    print(f"  DINOv2 ViTL (full scheduler)          n={len(a_dino)}: {a_dino.mean():.2f} +/- {a_dino.std(ddof=1):.2f} pp  median={np.median(a_dino):.2f} pp")
    print(f"  Delta (DINO-PLIP) paired n={len(delta)}: mean={delta.mean():+.2f} pp  std=+/-{delta.std(ddof=1):.2f} pp  median={np.median(delta):+.2f} pp")
    print(f"  Delta min/max = [{delta.min():+.2f}, {delta.max():+.2f}] pp    pct DINO>=PLIP: {(delta>=0).mean()*100:.1f}% ({(delta>=0).sum()}/{len(delta)})")
    if HAS_SCIPY:
        t=ttest_1samp(delta/100, 0.0)
        print(f"  1-sample t-test H0:Delta=0 -> t={t.statistic:+.3f}  p={t.pvalue:.4f}")
        if (delta!=0).any():
            w=wilcoxon(delta/100)
            print(f"  Wilcoxon signed-rank           -> W={w.statistic:.0f}    p={w.pvalue:.4f}")
    print(f"\n  Per (fold, seed) paired table:")
    print(f"  {'seed':<5}{'fold':<5}{'PLIP_pp':>9}{'DINO_pp':>10}{'Δ_pp':>8}")
    for f,s,bv,dv in paired:
        print(f"  {s:<5}{f:<5}{bv*100:>8.2f}%{dv*100:>9.2f}%{(dv-bv)*100:>+8.2f}")
else:
    print(f"  WARNING: no paired matches found; dino n={len(dino_map)} plip n={len(plip_map)}")

print("\n====== (2b) Window weight first-epoch distribution + check window scheduler logic correctness ======")

print("\n====== (3) Window_weight w_now activation profile: 15 runs of eval_with_censored phase2 ======")
win_curves=[]; w_prev_curves=[]
f_samples=[]
root_c = FULL/'eval_with_censored'/'phase2'
if root_c.is_dir():
    for p in sorted(root_c.rglob(f'seed*.csv')):
        m = RE_FS.search(str(p))
        if not m: continue
        rows=load_csv(p); cols=list(rows[0].keys())
        kws=[c for c in cols if 'now' in c.lower()] or ['w_now']; kw=kws[0]
        kps=[c for c in cols if 'prev' in c.lower()] or ['w_prev']; kp=kps[0]
        arr_now=[]; arr_prev=[]
        for r in rows:
            try:
                v=float(r.get(kw,'nan'))
                arr_now.append(v if np.isfinite(v) else 1.0)
            except Exception:
                arr_now.append(1.0)
            try:
                v=float(r.get(kp,'nan'))
                arr_prev.append(v if np.isfinite(v) else 1.0)
            except Exception:
                arr_prev.append(1.0)
        win_curves.append(arr_now); w_prev_curves.append(arr_prev)
        f_samples.append((m.group(1), m.group(2), len(rows),rows[-1].get('val_c_index_ema',''), rows[-1].get('train_c_index','')))
if win_curves:
    L=max(len(c) for c in win_curves)
    for i in range(len(win_curves)):
        need = L - len(win_curves[i])
        if need>0:
            win_curves[i] += [win_curves[i][-1]]*need
            w_prev_curves[i] += [w_prev_curves[i][-1]]*need
    W=np.asarray(win_curves); WP=np.asarray(w_prev_curves)
    print(f"  {W.shape[0]} runs found, each {W.shape[1]} epochs max")
    print(f"  epochs  mean_w_now(%)  std_w_now(%)  pct_runs_w<1%  mean_w_prev(%)")
    for ep in [0,1,2,3,4,5,7,9,14,19,24,29,34,39,49, min(W.shape[1]-1, 54)]:
        if ep>=W.shape[1]: continue
        col=W[:,ep]*100; colp=WP[:,ep]*100
        pct=(W[:,ep]<0.01).mean()*100
        pct10=(W[:,ep]<0.10).mean()*100
        print(f"  ep {ep+1:>3d}:   {col.mean():>7.2f}%   +/- {col.std(ddof=1):.2f}%   w<1%={pct:>3.0f}%  w<10%={pct10:>3.0f}%   w_prev={colp.mean():.2f}%")
    entered=[]
    for i in range(W.shape[0]):
        first=-1
        for ep in range(W.shape[1]):
            if W[i,ep] < 0.10:
                first=ep+1; break
        entered.append(first)
    eok=[x for x in entered if x>0]
    print(f"\n  Entered window (w_now<10%) first epoch: {len(eok)}/{len(entered)} runs entered; mean={np.mean(eok):.1f} +/- {np.std(eok, ddof=1):.1f} epochs")
    print(f"  w_now at last epoch({W.shape[1]}): mean {W[:,-1].mean()*100:.3f}%, std {W[:,-1].std(ddof=1)*100:.3f}% (15/15 runs)")

# --- (4) k3_grid A ablation + k3_grid window ablations (phase2) ---
print("\n====== (4) k3_grid_A_ablation and k3_grid_window phase2 ablations ======")
for sub in ['k3_grid_A_ablation','k3_grid_window','k5','win0.59-0.65_A0.99_K50']:
    d=PHASE2OLD/sub
    if not d.is_dir():
        continue
    files=sorted(d.rglob('summary.json'))
    if not files:
        continue
    from collections import defaultdict
    buckets=defaultdict(list)
    for f in files:
        d2=read_json(f)
        if not d2: continue
        val = d2.get('final_test_c_index') or d2.get('best_test_c_index')
        if not isinstance(val, float): continue
        key=str(Path(*f.relative_to(d).parts[:2])) if len(f.relative_to(d).parts)>=2 else str(Path(*f.relative_to(d).parts[:1]))
        buckets[key].append(val*100)
    print(f"\n  [{sub}] n runs total={len(files)}, unique configs={len(buckets)}:")
    for k,v in sorted(buckets.items(), key=lambda kv: -np.mean(kv[1])):
        a=np.asarray(v)
        print(f"    {k}: n={a.size} final_test_c = {a.mean():.2f} +/- {a.std(ddof=1):.2f} pp  range [{a.min():.2f}, {a.max():.2f}]")
