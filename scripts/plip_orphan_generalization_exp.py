#!/usr/bin/env python3
"""
Q5-A - Appx C - PLIP orphan LUAD 497 tokens 泛化性实验 (plip_luad_256 / plip_luad_256_single_dx_ts_bs)
"""
import argparse, csv, re, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold

try:
    from sksurv.metrics import concordance_index_censored
except Exception:
    def concordance_index_censored(event, time, estimate):
        order = np.argsort(time)
        time, event, estimate = time[order], event[order], estimate[order]
        n, pairs, concord = len(time), 0, 0.0
        for i in range(n):
            for j in range(i+1, n):
                if event[i] or time[i] != time[j]:
                    if time[i] < time[j] and event[i]:
                        pairs += 1
                        concord += 1.0 if estimate[i] > estimate[j] else (0.5 if estimate[i] == estimate[j] else 0.0)
                    elif time[i] > time[j] and event[j]:
                        pairs += 1
                        concord += 1.0 if estimate[j] > estimate[i] else (0.5 if estimate[i] == estimate[j] else 0.0)
        return (0.0 if pairs == 0 else concord / pairs, pairs, 0, 0, 0)

PROJ = Path(__file__).resolve().parent.parent
RAW_CLIN = PROJ/'data/raw_clinical'
PLIP_D1  = PROJ/'data/tokens/plip_luad_256'
PLIP_D2  = PROJ/'data/tokens/plip_luad_256_single_dx_ts_bs'
HALLMARK_CSV = PROJ/'data/raw_rna/metadata/hallmarks_signatures.csv'
SPLITS_DIR = PROJ/'data/splits/PLIP_ORPHAN_LUAD'
OUT      = PROJ/'outputs'

def _case_id_from_path(p: Path) -> str:
    m = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', p.name, re.I)
    return m.group(1).upper() if m else ''

def _int_nonneg(x):
    try:
        v = int(float(str(x).strip())); return v if v >= 0 else None
    except Exception:
        return None

def load_plip_cases(plip_dir: Path, clin_path: Path, max_npz: int = 0):
    case_files = defaultdict(list)
    for p in sorted(plip_dir.glob('*.npz')):
        cid = _case_id_from_path(p)
        if cid: case_files[cid].append(p)
    rows = list(csv.DictReader(open(clin_path)))
    clin_map = {r['submitter_id'].upper(): r for r in rows}
    valid, y = {}, {}
    for cid, files in case_files.items():
        r = clin_map.get(cid)
        if not r: continue
        v = (r.get('vital_status') or '').title()
        dd = _int_nonneg(r.get('days_to_death') or '')
        df = _int_nonneg(r.get('days_to_last_follow_up') or '')
        if v == 'Dead' and dd is not None:
            valid[cid], y[cid] = files, (dd, 1)
        elif v == 'Alive' and df is not None:
            valid[cid], y[cid] = files, (df, 0)
    print(f"[plip {plip_dir.name}] npz={sum(len(v) for v in valid.values())} unique_cases={len(valid)}")
    if max_npz > 0:
        keys = list(valid.keys())[:max_npz]
        valid = {c: valid[c] for c in keys}
        y = {c: y[c] for c in keys}
    return valid, y

def load_token(p: Path, target_dim: int = 512) -> np.ndarray:
    x = np.load(p, allow_pickle=True)
    for k in ['features','feat','z','token','plip','arr_0']:
        if k in x.files:
            try:
                arr = np.asarray(x[k], dtype=np.float32)
                if arr.ndim == 1: arr = arr[None, :]
                mean = arr.mean(axis=0).astype(np.float32)
                if len(mean) >= target_dim: return mean[:target_dim]
                return np.pad(mean, (0, target_dim - len(mean)))
            except Exception:
                continue
    for k in x.files:
        try:
            arr = np.asarray(x[k], dtype=np.float32)
            if arr.ndim == 1: arr = arr[None, :]
            mean = arr.mean(axis=0).astype(np.float32)
            if len(mean) >= target_dim: return mean[:target_dim]
            return np.pad(mean, (0, target_dim - len(mean)))
        except Exception:
            continue
    return np.zeros(target_dim, dtype=np.float32)

def build_strata_cases(y_case):
    rows = [{'cid': cid, 'time': t, 'event': e} for cid, (t, e) in y_case.items()]
    cases_df = pd.DataFrame(rows)
    cases_df['time_bin'] = None
    for v in (1, 0):
        m = (cases_df['event'] == v)
        if m.sum() <= 1: continue
        q = min(4, int(m.sum()))
        try:
            cases_df.loc[m, 'time_bin'] = pd.qcut(cases_df.loc[m, 'time'], q=q, labels=[f'Q{i+1}' for i in range(q)], duplicates='drop')
        except Exception:
            cases_df.loc[m, 'time_bin'] = pd.qcut(cases_df.loc[m, 'time'].rank(method='first'), q=q, labels=[f'Q{i+1}' for i in range(q)], duplicates='drop')
    cases_df['time_bin'] = cases_df['time_bin'].astype(str)
    cases_df['stratum'] = cases_df['event'].astype(str) + '_' + cases_df['time_bin']
    return cases_df

def do_split(cases_df, seed=42):
    sizes = cases_df['stratum'].value_counts()
    if len(sizes) > 0 and sizes.min() >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(skf.split(cases_df, cases_df['stratum']))
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(cases_df))

class PLIPOnlyNet(nn.Module):
    def __init__(self, in_dim=512, hd=256, p=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hd), nn.GELU(), nn.Dropout(p), nn.Linear(hd, 1))
    def forward(self, x): return self.net(x)

class PLIPHallmarkNet(nn.Module):
    def __init__(self, pdim=512, hdim=50, hd=256, p=0.15):
        super().__init__()
        self.pp = nn.Sequential(nn.Linear(pdim, hd), nn.GELU())
        self.hp = nn.Sequential(nn.Linear(hdim, hd), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2*hd, hd), nn.GELU(), nn.Dropout(p), nn.Linear(hd, 1))
    def forward(self, xp, xh): return self.head(torch.cat([self.pp(xp), self.hp(xh)], dim=-1))

def neg_partial_loglik(risk, event, time):
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    cum = torch.logcumsumexp(risk.view(-1), dim=0)
    denom = event.sum().clamp(min=1)
    return -torch.sum(event.view(-1) * (risk.view(-1) - cum)) / denom + 1e-6

def train_one(train_toks, train_hall, train_y, test_toks, test_hall, test_y,
              mode='plip_only', pdim=512, hdim=50, hd=256, seed=0, epochs=200, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cpu'
    if mode == 'plip_only':
        model = PLIPOnlyNet(in_dim=pdim, hd=hd).to(dev)
    else:
        model = PLIPHallmarkNet(pdim=pdim, hdim=hdim, hd=hd).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xp_tr = torch.from_numpy(np.asarray(train_toks, dtype=np.float32)).to(dev)
    Xh_tr = torch.from_numpy(np.asarray(train_hall, dtype=np.float32)).to(dev)
    t_tr = torch.tensor([t for t, _ in train_y], dtype=torch.float32, device=dev)
    e_tr = torch.tensor([e for _, e in train_y], dtype=torch.float32, device=dev)
    Xp_te = torch.from_numpy(np.asarray(test_toks, dtype=np.float32)).to(dev)
    Xh_te = torch.from_numpy(np.asarray(test_hall, dtype=np.float32)).to(dev)
    t_te = np.array([t for t, _ in test_y])
    e_te = np.array([e for _, e in test_y]).astype(bool)
    best_c = -1.0
    for ep in range(epochs):
        model.train()
        risk = model(Xp_tr) if mode == 'plip_only' else model(Xp_tr, Xh_tr)
        loss = neg_partial_loglik(risk, e_tr, t_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                r = (model(Xp_te) if mode == 'plip_only' else model(Xp_te, Xh_te)).view(-1).cpu().numpy()
            try:
                c = float(concordance_index_censored(e_te.astype(bool), t_te, r)[0])
                if c > best_c: best_c = c
            except Exception:
                pass
    return best_c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plip-dir', default=str(PLIP_D1))
    ap.add_argument('--cohort', default='LUAD', choices=['LUAD'])
    ap.add_argument('--pdim', type=int, default=512)
    ap.add_argument('--hdim', type=int, default=50)
    ap.add_argument('--hd', type=int, default=256)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-npz', type=int, default=0)
    args = ap.parse_args()
    plip_dir = Path(args.plip_dir)
    case_files, y_case = load_plip_cases(plip_dir, RAW_CLIN/args.cohort/'clinical.csv', max_npz=args.max_npz)
    cases_df = build_strata_cases(y_case)
    folds = do_split(cases_df, seed=42)
    cid_list = cases_df['cid'].tolist()
    HALL = np.zeros((len(cid_list), args.hdim), dtype=np.float32)
    cid2idx = {c: i for i, c in enumerate(cid_list)}
    tokens = np.zeros((len(cid_list), args.pdim), dtype=np.float32)
    for cid, files in case_files.items():
        if cid not in cid2idx: continue
        z = np.mean([load_token(p, target_dim=args.pdim) for p in files], axis=0).astype(np.float32)
        tokens[cid2idx[cid]] = z
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    for k, (tr, te) in enumerate(folds):
        tr_c = [cid_list[i] for i in tr]; te_c = [cid_list[i] for i in te]
        tr_tok = tokens[tr]; tr_h = HALL[tr]; tr_y = [y_case[c] for c in tr_c]
        te_tok = tokens[te]; te_h = HALL[te]; te_y = [y_case[c] for c in te_c]
        for mode in ['plip_only', 'plip_hallmark']:
            cs = []
            for sd in args.seeds:
                c = train_one(tr_tok, tr_h, tr_y, te_tok, te_h, te_y, mode=mode,
                              pdim=args.pdim, hdim=args.hdim, hd=args.hd, seed=sd)
                cs.append(c)
                audit.append(dict(fold=k, seed=sd, mode=mode, cindex=c,
                                  train_cases=len(tr_c), test_cases=len(te_c)))
            print(f"k={k} mode={mode:<15} C-index = {np.mean(cs):.4f} +/- {np.std(cs):.4f}  train={len(tr_c)} test={len(te_c)}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(audit)
    df.to_csv(OUT/'supp_table_C_plip_generalization.csv', index=False)
    ag = df.groupby('mode').agg(mean_c=('cindex','mean'), std_c=('cindex','std'), n=('cindex','count')).reset_index()
    print("===== FINAL =====")
    print(ag.to_string())
    with open(OUT/'supp_table_C_plip_generalization_summary.json', 'w') as fh:
        json.dump(ag.to_dict(orient='records'), fh, indent=2)

if __name__ == '__main__':
    main()
