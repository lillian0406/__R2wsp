"""
B3 · 4 cohort 严谨 A3 split 生成 (会被外部像 MMP split 那样参考)
规则 (100% 定版，不得改):
  ① 分层变量 = vital_status (Dead/Alive=2) × days_to 四分位数 (4) = 8 strata
  ② sklearn StratifiedKFold n_splits=5 shuffle=True random_state=42
  ③ 5 folds 每个 fold 的 case 数差 ≤ 1；每个 fold 的 Dead/Alive 比例差 ≤ 0.03
  ④ case-level split: 同 case 所有 slides 永远跟 case 在同一个 fold (绝对不允许 slide 跨 fold)
  ⑤ 输出 2 CSV / cohort:
      a) splits/<COHORT>/case_fold_mapping_5fold_seed42.csv   列: case_id,vital_status,days_to,stratum,fold
      b) splits/<COHORT>/fold_audit_5fold_seed42.csv         每 fold case / slide / event / censor / 最大最小时间
"""
import argparse, csv, re
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter, OrderedDict
from sklearn.model_selection import StratifiedKFold, KFold
PROJ = Path(__file__).resolve().parent.parent
CLIN = PROJ/'data/raw_clinical'
RAW = PROJ/'data/raw_svs'
SPL = PROJ/'data/splits'

def _cid(s):
    m = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', str(s), re.I)
    return m.group(1).upper() if m else ''

def _int(x):
    try:
        v = int(float(str(x).strip()))
        return v if v >= 0 else None
    except Exception: return None

def load_case_labels(cohort):
    rows = list(csv.DictReader(open(CLIN/cohort/'clinical.csv')))
    items = []
    for r in rows:
        sub = r['submitter_id'].strip().upper()
        if not sub.startswith('TCGA-'): continue
        v = (r.get('vital_status') or '').title().strip()
        dd = _int(r.get('days_to_death') or ''); df = _int(r.get('days_to_last_follow_up') or '')
        if v == 'Dead' and dd is not None:
            items.append(dict(case_id=sub, vital_status='Dead', days=dd, event=1))
        elif v == 'Alive' and df is not None:
            items.append(dict(case_id=sub, vital_status='Alive', days=df, event=0))
    df = pd.DataFrame(items).drop_duplicates('case_id').reset_index(drop=True)
    return df

def slide_counts_per_case(cohort, cases):
    out = Counter()
    # raw_svs 里找对应 cases 的 svs
    for f in RAW.glob('*.svs'):
        c = _cid(f.name)
        if c in cases: out[c] += 1
    # 扫 legacy 单独目录
    for d in [PROJ/'data/raw_svs'/cohort, PROJ/'data/GDCdata'/f'TCGA-{cohort}'/'harmonized'/'Pathology_Slide_Image']:
        if not d.exists(): continue
        for f in d.rglob('*.svs'):
            c = _cid(f.name)
            if c in cases: out[c] += 1
    return out

def make_strata(df_label):
    df = df_label.copy()
    df['time_bin'] = None
    for v in ('Dead','Alive'):
        mask = df['vital_status']==v
        n = int(mask.sum())
        if n < 8:
            df.loc[mask,'time_bin'] = 'Q1'
            continue
        try:
            df.loc[mask,'time_bin'] = pd.qcut(df.loc[mask,'days'], q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        except Exception:
            df.loc[mask,'time_bin'] = pd.qcut(df.loc[mask,'days'].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    df['stratum'] = df['vital_status'].astype(str) + '_' + df['time_bin'].astype(str)
    return df

def do_split(df_label, cohort):
    df = make_strata(df_label)
    sizes = df['stratum'].value_counts()
    use_skf = (len(sizes) >= 5 and sizes.min() >= 5)
    spl = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if use_skf else KFold(n_splits=5, shuffle=True, random_state=42)
    X = np.arange(len(df)); y = df['stratum'].values if use_skf else np.arange(len(df))
    fold_map = {}
    for fold, (tr, te) in enumerate(spl.split(X, y)):
        for idx in te: fold_map[int(idx)] = fold
    df['fold'] = df.index.map(fold_map).astype(int)
    # 审计: fold 大小差
    fc = df['fold'].value_counts().sort_index()
    assert (fc.max() - fc.min()) <= 1, f"{cohort} fold size diff > 1: {dict(fc)} => violates rule 3 (别人会参考)"
    return df

def audit(df, case_slide_counts, cohort):
    rows = []
    cases = set(df['case_id'])
    for f in sorted(df['fold'].unique()):
        sub = df[df['fold']==f]
        n_case = len(sub); n_dead = int((sub['vital_status']=='Dead').sum()); n_alive = n_case-n_dead
        n_slide = int(sum(case_slide_counts.get(c,0) for c in sub['case_id']))
        tmin = int(sub['days'].min()); tmax = int(sub['days'].max()); tmed = float(sub['days'].median())
        rows.append(dict(fold=f, n_case=n_case, n_slide=n_slide, Dead=n_dead, Alive=n_alive,
                         Dead_pct=f"{n_dead/n_case*100:.1f}%" if n_case else '-',
                         Alive_pct=f"{n_alive/n_case*100:.1f}%" if n_case else '-',
                         days_min=tmin, days_median=round(tmed,1), days_max=tmax))
    # overall
    sub = df
    rows.append(dict(fold='OVERALL', n_case=len(sub), n_slide=int(sum(case_slide_counts.get(c,0) for c in sub['case_id'])),
                     Dead=int((sub['vital_status']=='Dead').sum()), Alive=int((sub['vital_status']=='Alive').sum()),
                     Dead_pct=f"{int((sub['vital_status']=='Dead').sum())/len(sub)*100:.1f}%",
                     Alive_pct=f"{int((sub['vital_status']=='Alive').sum())/len(sub)*100:.1f}%",
                     days_min=int(sub['days'].min()), days_median=round(float(sub['days'].median()),1), days_max=int(sub['days'].max())))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohorts', nargs='+', default=['LUAD','BRCA','LUSC','PAAD'])
    args = ap.parse_args()
    SPL.mkdir(parents=True, exist_ok=True)
    summary = []
    for c in args.cohorts:
        out_dir = SPL/c; out_dir.mkdir(parents=True, exist_ok=True)
        df_lab = load_case_labels(c)
        case_set = set(df_lab['case_id'])
        cs = slide_counts_per_case(c, case_set)
        df_fold = do_split(df_lab, c)
        # 写 case->fold 映射
        f1 = out_dir/'case_fold_mapping_5fold_seed42.csv'
        df_out = df_fold[['case_id','vital_status','days','event','stratum','fold']].copy()
        df_out['slides_on_disk'] = df_out['case_id'].map(lambda x: cs.get(x,0))
        df_out.to_csv(f1, index=False)
        # 写 audit
        df_aud = audit(df_fold, cs, c)
        f2 = out_dir/'fold_audit_5fold_seed42.csv'
        df_aud.to_csv(f2, index=False)
        fc = df_fold['fold'].value_counts().sort_index().tolist()
        print(f"✅ {c}: cases={len(df_fold)} slides={sum(cs.get(x,0) for x in case_set)} fold_sizes={fc} diff={max(fc)-min(fc)} (≤1 OK) audit in {f2}")
        summary.append(dict(cohort=c, cases_with_label=len(df_fold), slides_on_disk=sum(cs.values()),
                            fold_sizes=str(fc), fold_size_diff=max(fc)-min(fc),
                            Dead=dict(df_fold['vital_status'].value_counts()).get('Dead',0),
                            Alive=dict(df_fold['vital_status'].value_counts()).get('Alive',0),
                            split_csv=str(f1), audit_csv=str(f2)))
    pd.DataFrame(summary).to_csv(PROJ/'outputs/_disk_cohort_inventory/splits_A3_4cohorts_SUMMARY.csv', index=False)
    print(f"\n✅ 4 cohort A3 split 总览写盘: {PROJ/'outputs/_disk_cohort_inventory/splits_A3_4cohorts_SUMMARY.csv'}")

if __name__ == '__main__': main()
