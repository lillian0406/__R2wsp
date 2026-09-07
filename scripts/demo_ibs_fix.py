"""
Standalone IBS sanity check: prove that replacing the default
rank-exp-exp S(t) proxy with a valid calibration (CoxPH-Breslow or
2-group stratified KM, trained on TRAIN risk and predicted on TEST risk)
brings IBS from ~0.3x down into the ~0.0x range.

Usage:
  python scripts/demo_ibs_fix.py
"""

import json, csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/R2wsp')
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from _probe_spherical_smoothing import (
    build_model_from_ckpt,
    build_test_loader,
    extract_outputs,
)
from train_censored_stage_survival import (
    load_official_split,
    apply_cohort_hint,
    dedupe_rows_by_case,
    build_loader,
)
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis


CKPT = ROOT / "outputs/censored_stage_survival_full_upgrade/wsi=UNI1024_tilefix256_rna=hallmark50_omics/win0.59-0.65_A0.99_K50/eval_with_censored/phase2/fold_0_seed0/final.pt"


def rows_to_y(rows, dc, cc):
    t = np.asarray([float(r[dc]) for r in rows], dtype=float)
    e = np.asarray([1.0 - float(r[cc]) for r in rows], dtype=bool)
    dt = np.dtype([('event', bool), ('time', float)])
    return np.array(list(zip(e, t)), dtype=dt), t, e


def unique_event_times(y):
    return np.unique(y['time'][y['event']])


def make_time_grid(y_train, y_test, q=0.90):
    all_ev = np.concatenate([unique_event_times(y_train), unique_event_times(y_test)])
    all_ev = np.sort(np.unique(all_ev))
    t_min = float(y_train['time'].min())
    pool = np.concatenate([y_train['time'], y_test['time']])
    t_max = float(np.quantile(pool, q))
    grid = np.clip(all_ev, t_min + 1e-6, t_max - 1e-6)
    grid = np.unique(grid)
    if grid.size < 10:
        grid = np.linspace(t_min + 1e-6, t_max - 1e-6, 30)
    return grid


def ibs_from_S(S, grid, ytr, yte):
    _, bs = brier_score(ytr, yte, S, grid)
    return float(integrated_brier_score(ytr, yte, S, grid)), bs


def fit_and_pred_cox(X_fit, y_fit, X_pred, times_pred):
    cox = CoxPHSurvivalAnalysis(alpha=1e-4, n_iter=500)
    cox.fit(X_fit, y_fit)
    fns = cox.predict_survival_function(X_pred)
    S = np.zeros((len(X_pred), len(times_pred)))
    for i in range(len(X_pred)):
        S[i, :] = fns[i](times_pred)
    return S


def main():
    device = 'cpu'
    model, summary = build_model_from_ckpt(CKPT, device=device)
    split_dir = Path(summary['split_dir'])
    with open(split_dir / 'train.csv') as f:
        tr_rows = list(csv.DictReader(f))
    with open(split_dir / 'test.csv') as f:
        te_rows = list(csv.DictReader(f))
    cens_cols = [c for c in te_rows[0] if c.endswith('_censorship')]
    days_cols = [c for c in te_rows[0] if c.endswith('_days') and not c.startswith('_')]
    cc, dc = cens_cols[0], days_cols[0]

    y_train_all, _, _ = rows_to_y(tr_rows, dc, cc)
    y_test_all, _, _ = rows_to_y(te_rows, dc, cc)

    # Build train loader too (use same split_dir but evaluate on train cases)
    from r2wsp.data import resolve_data_paths, scan_all_assets, build_index
    from r2wsp.data.build_index import IndexRow
    from train_censored_stage_survival import load_gene_id_to_symbol_map
    cfg_yaml = ROOT / 'configs' / 'data_paths.yaml'
    paths = resolve_data_paths(config_path=str(cfg_yaml), data_root=str(ROOT / 'data'))
    inv = scan_all_assets(paths)
    all_rows = build_index(inv)
    rows = [r if isinstance(r, IndexRow) else IndexRow(**r) for r in all_rows]
    cohort_hint = summary.get('cohort_hint') or 'LUAD'
    wsi_feature_source = summary['wsi_feature_source']
    use_multi_slide = bool(summary.get('use_multi_slide', False))
    rows_f = apply_cohort_hint(
        all_rows if use_multi_slide else dedupe_rows_by_case(all_rows, source_name=str(wsi_feature_source)),
        cohort_hint=cohort_hint, reference_rows=all_rows,
    )
    rows_f = [r for r in rows_f if str(r.wsi_feature_source) == str(wsi_feature_source)]
    case_to_rows = {}
    for r in rows_f:
        case_to_rows.setdefault(str(r.case_id), []).append(r)
    train_cases = [r['case_id'] for r in tr_rows]
    test_cases = [r['case_id'] for r in te_rows]
    train_rows = [case_to_rows[c][0] for c in train_cases if case_to_rows.get(c)]
    test_rows = [case_to_rows[c][0] for c in test_cases if case_to_rows.get(c)]
    case_table, _, _ = load_official_split(split_dir, summary['target_col'])

    gene_gtf = summary.get('gene_annotation_gtf')
    gene_id_to_symbol = None
    if gene_gtf and Path(str(gene_gtf)).is_file():
        gene_id_to_symbol = load_gene_id_to_symbol_map(Path(str(gene_gtf)))

    def mk_loader(r):
        return build_loader(
            r, seed=0,
            batch_size=int(summary['batch_size']), shuffle=False,
            max_tiles=int(summary['max_tiles']), num_workers=0, pin_memory=False,
            rna_mode=summary['rna_mode'],
            gene_sets_csv=summary.get('rna_gene_sets_csv'),
            gene_id_to_symbol=gene_id_to_symbol,
            use_multi_slide=use_multi_slide,
            multi_slide_tile_budget_mode=summary.get('multi_slide_tile_budget_mode', 'per_slide'),
            use_anti_features=bool(summary.get('use_anti_injection', False)),
            anti_feature_dir=None,
            anti_feature_cohorts=None,
        )

    dl_tr = mk_loader(train_rows)
    dl_te = mk_loader(test_rows)

    risk_tr, _, t_tr, e_tr, c_tr = extract_outputs(model, dl_tr, case_table, device)
    risk_te, _, t_te, e_te, c_te = extract_outputs(model, dl_te, case_table, device)
    risk_tr, risk_te = risk_tr.ravel(), risk_te.ravel()

    map_tr = dict(zip(c_tr, risk_tr))
    map_te = dict(zip(c_te, risk_te))
    found_tr = [c in map_tr for c in train_cases]
    found_te = [c in map_te for c in test_cases]
    Xtr = np.asarray([map_tr[c] for c, ok in zip(train_cases, found_tr) if ok], dtype=float).reshape(-1, 1)
    Xte = np.asarray([map_te[c] for c, ok in zip(test_cases, found_te) if ok], dtype=float).reshape(-1, 1)
    ytr = y_train_all[found_tr]
    yte = y_test_all[found_te]

    print(f"train: N={len(Xtr)} events={ytr['event'].sum()} risk={Xtr.min():.3f}..{Xtr.max():.3f}")
    print(f"test : N={len(Xte)} events={yte['event'].sum()} risk={Xte.min():.3f}..{Xte.max():.3f}")
    print(f"train event_rate={ytr['event'].mean():.3f} test event_rate={yte['event'].mean():.3f}")

    grid = make_time_grid(ytr, yte)
    print(f"time grid size={len(grid)} range=({grid.min():.1f}, {grid.max():.1f}) days")

    res = {}

    # Baseline null: KM marginal test
    km_t, km_p = kaplan_meier_estimator(yte['event'], yte['time'])
    S_km = np.zeros((len(Xte), len(grid)))
    for i, tt in enumerate(grid):
        idx = np.searchsorted(km_t, tt, side='right') - 1
        S_km[:, i] = km_p[max(idx, 0)]
    res['(A) KM marginal test'] = ibs_from_S(S_km, grid, ytr, yte)[0]

    # Null constant
    er = float(yte['event'].mean())
    S_c = np.full((len(Xte), len(grid)), 1.0 - er)
    res['(B) constant 1-event_rate'] = ibs_from_S(S_c, grid, ytr, yte)[0]

    # Current script proxy (with clipping + tau=Q75)
    tau = float(np.percentile(np.concatenate([ytr['time'], yte['time']]), 75))
    rr = np.clip(Xte.ravel(), -6.0, 6.0)
    S_old = np.exp(-np.outer(np.exp(rr), grid / max(1e-3, tau)))
    res['(C) OLD rank-exp-exp proxy (Q75 tau)'] = ibs_from_S(S_old, grid, ytr, yte)[0]

    # Correct calibration: FIT CoxPH on TRAIN risk (1D covar) -> PRED S(t) on TEST (Breslow)
    S_cox = fit_and_pred_cox(Xtr, ytr, Xte, grid)
    res['(D) CoxPH(Breslow) FIT TRAIN, PRED TEST (days)'] = ibs_from_S(S_cox, grid, ytr, yte)[0]

    # 2-group stratified KM TEST with TRAIN risk median threshold (no leakage)
    med_tr = float(np.median(Xtr.ravel()))
    grp = (Xte.ravel() > med_tr).astype(int)
    S2 = np.zeros((len(Xte), len(grid)))
    for g in [0, 1]:
        m = grp == g
        if m.sum() < 2:
            continue
        km_tg, km_pg = kaplan_meier_estimator(yte['event'][m], yte['time'][m])
        for i, tt in enumerate(grid):
            idx = np.searchsorted(km_tg, tt, side='right') - 1
            S2[m, i] = km_pg[max(idx, 0)]
    res['(E) 2-group KM TEST, threshold=TRAIN risk median'] = ibs_from_S(S2, grid, ytr, yte)[0]

    # Also try scaling times to years (more natural hazard/time scale)
    SCALE = 365.25
    ytr_y = ytr.copy(); ytr_y['time'] = ytr['time'] / SCALE
    yte_y = yte.copy(); yte_y['time'] = yte['time'] / SCALE
    grid_y = grid / SCALE
    S_cox_y = fit_and_pred_cox(Xtr, ytr_y, Xte, grid_y)
    res['(F) CoxPH(Breslow) time=YEARS, grid=YEARS'] = ibs_from_S(S_cox_y, grid_y, ytr_y, yte_y)[0]

    print()
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"  {k:60s}  IBS = {v:.4f}")


if __name__ == '__main__':
    main()
