#!/usr/bin/env python3
import csv, re
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold
PROJ = Path(__file__).resolve().parent.parent
CLIN = PROJ/'data/raw_clinical'
RAW = PROJ/'data/raw_svs'
SPL = PROJ/'data/splits'

def _cid(s):
    m = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', str(s), re.I)
    return m.group(1).upper() if m else ''

def _to_int(x):
    try:
        s = str(x).strip()
        if not s or s.lower() in ('--','na','nan','none','not applicable'): return None
        v = int(float(s))
        return v if v >= 0 else None
    except Exception: return None

def _norm_vs(x):
    s = (str(x) or '').strip().title()
    if s.startswith('Dead') or s in ('Dead','Died','Deceased'): return 'Dead'
    if s.startswith('Alive') or s in ('Alive','Living','Not Dead'): return 'Alive'
    return ''

def load_labels(cohort):
    p = CLIN/cohort/'clinical.csv'
    rows = [{k.strip(): (str(v) if v is not None else '') for k,v in r.items()} for r in csv.DictReader(open(p, newline='', encoding='utf-8-sig'))]
    items = []
    for r in rows:
        sub = ''
        for k in ['submitter_id','case_id','case_submitter_id','bcr_patient_barcode']:
            if str(r.get(k,'')).upper().startswith('TCGA-'): sub = str(r[k]).strip().upper(); break
        if not sub: continue
        vs = _norm_vs(r.get('vital_status',''))
        dtd = _to_int(r.get('days_to_death','')); dtf = _to_int(r.get('days_to_last_follow_up',''))
        if vs == 'Dead' and dtd is not None:
            items.append(dict(case_id=sub, vital_status='Dead', days=dtd, event=1, label_complete=True))
        elif vs == 'Alive' and dtf is not None:
            items.append(dict(case_id=sub, vital_status='Alive', days=dtf, event=0, label_complete=True))
        elif vs in ('Dead','Alive'):
            # 标签不全（Alive 空 follow up / Dead 空 dtd）：留作 fold 分配，但 include_in_eval=False（MMP 式严谨）
            med = 365*3 if vs == 'Dead' else 365*5
            items.append(dict(case_id=sub, vital_status=vs, days=med, event=(1 if vs=='Dead' else 0), label_complete=False))
    df = pd.DataFrame(items).drop_duplicates('case_id').reset_index(drop=True)
    comp = int(df['label_complete'].sum())
    print(f"  {cohort}: cases_total_vital={len(df)} label_complete(event+time 都有)={comp} ({dict(df.loc[df['label_complete'],'vital_status'].value_counts())}); 不完整={len(df)-comp} —— 最终用户给的公理是 total alive+dead 应该达 LUAD 522/BRCA 537/LUSC 504/PAAD 185（就是 total vital non空）")
    return df

def slide_counts(cohort, case_set):
    out = Counter()
    for f in RAW.glob('*.svs'):
        c = _cid(f.name);
        if c in case_set: out[c]+=1
    return out

def build_strata(df):
    x = df.copy(); x['time_bin']=None
    for v in ('Dead','Alive'):
        m = x['vital_status']==v; n=int(m.sum())
        if n < 4: x.loc[m,'time_bin']='Q1'; continue
        try: x.loc[m,'time_bin'] = pd.qcut(x.loc[m,'days'], q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        except: x.loc[m,'time_bin'] = pd.qcut(x.loc[m,'days'].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    x['stratum'] = x['vital_status'].astype(str) + '_' + x['time_bin'].astype(str)
    return x

def split(df, cohort):
    x = build_strata(df); sz = x['stratum'].value_counts()
    use_skf = (len(sz) >= 5 and sz.min() >= 5)
    sp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if use_skf else KFold(n_splits=5, shuffle=True, random_state=42)
    X = np.arange(len(x)); y = x['stratum'].values if use_skf else np.arange(len(x))
    fm = {}
    for f,(tr,te) in enumerate(sp.split(X,y)):
        for i in te: fm[int(i)] = f
    x['fold'] = x.index.map(fm).astype(int)
    fc = x['fold'].value_counts().sort_index().tolist()
    assert (max(fc)-min(fc)) <= 1, f"{cohort} fold diff > 1: {fc}"
    return x

def audit(df, cs):
    rs = []
    for f in sorted(df['fold'].unique()):
        s = df[df['fold']==f]; nc = len(s)
        nd = int((s['vital_status']=='Dead').sum()); na = nc-nd; ns = int(sum(cs.get(c,0) for c in s['case_id']))
        ncomp = int(s['label_complete'].sum())
        rs.append(dict(fold=f, n_case=nc, n_slide=ns, Dead=nd, Alive=na, label_complete=ncomp,
                       Dead_pct=f"{nd/nc*100:.1f}%" if nc else '-', Alive_pct=f"{na/nc*100:.1f}%" if nc else '-'))
    s = df
    rs.append(dict(fold='OVERALL', n_case=len(s), n_slide=int(sum(cs.get(c,0) for c in s['case_id'])),
                   Dead=int((s['vital_status']=='Dead').sum()), Alive=int((s['vital_status']=='Alive').sum()),
                   label_complete=int(s['label_complete'].sum()),
                   Dead_pct=f"{int((s['vital_status']=='Dead').sum())/len(s)*100:.1f}%",
                   Alive_pct=f"{int((s['vital_status']=='Alive').sum())/len(s)*100:.1f}%"))
    return pd.DataFrame(rs)

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--cohorts', nargs='+', default=['LUAD','BRCA','LUSC','PAAD'])
    a = ap.parse_args(); SPL.mkdir(parents=True, exist_ok=True); summ = []
    for c in a.cohorts:
        out_dir = SPL/c; out_dir.mkdir(parents=True, exist_ok=True)
        lab = load_labels(c)
        cs = slide_counts(c, set(lab['case_id']))
        fold = split(lab, c)
        df_out = fold[['case_id','vital_status','days','event','stratum','label_complete','fold']].copy()
        df_out['slides_on_disk'] = df_out['case_id'].map(lambda x: cs.get(x,0))
        f1 = out_dir/'case_fold_mapping_5fold_seed42.csv'; df_out.to_csv(f1, index=False)
        aud = audit(fold, cs); f2 = out_dir/'fold_audit_5fold_seed42.csv'; aud.to_csv(f2, index=False)
        fc = fold['fold'].value_counts().sort_index().tolist()
        print(f"✅ {c}: total cases(vital non空)={len(fold)} label_complete={int(fold['label_complete'].sum())} slides_on_disk={sum(cs.values())} fold_sizes={fc} diff={max(fc)-min(fc)} ≤1 OK")
        summ.append(dict(cohort=c, cases_vital_nonnull=len(fold), cases_label_complete=int(fold['label_complete'].sum()),
                         slides_on_disk=sum(cs.values()), fold_sizes=str(fc), fold_size_diff=max(fc)-min(fc),
                         Dead=dict(fold['vital_status'].value_counts()).get('Dead',0),
                         Alive=dict(fold['vital_status'].value_counts()).get('Alive',0),
                         split_csv=str(f1), audit_csv=str(f2)))
    pd.DataFrame(summ).to_csv(PROJ/'outputs/_disk_cohort_inventory/splits_A3_4cohorts_SUMMARY.csv', index=False)
    print(f"✅ 总览写盘: {PROJ/'outputs/_disk_cohort_inventory/splits_A3_4cohorts_SUMMARY.csv'}")
if __name__ == '__main__': main()
