"""
Validate that the orthogonal decomposition gain is NOT an artifact of
systematically shifting censored patients.

Zero-impact, CPU only, reuses DINOv2 ViTL eval_with_censored final.pt ckpts.
Tests:
  (1) Event-stratified C-index delta: delta(c_uncensored) vs delta(c_censored).
      If gain comes mostly from the uncensored pair-contributions (ground truth labels)
      and not from censored pair-contributions, then the gain is genuine.
  (2) Risk L1 delta |r_new - r_orig| grouped by event: large on censored would be suspicious.
  (3) Per-fold censor rate vs delta C correlation: if positive, "more censored -> more gain" = suspicious.
  (4) Absolute rank-shift histogram grouped by event.
"""
import os, sys, json, re
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/R2wsp')
sys.path.insert(0, str(ROOT))

from scripts._probe_head_orthog_gated import (
    build_model_from_ckpt, build_test_loader,
    gated_orthog_decompose_risk, c_index,
    extract_fused_and_backbone,
)


def c_index_on_pairs(risk, events, times, pair_mask_event=None):
    """Compute standard c-index but restrict which (i,j) pairs contribute.
    pair_mask_event: (N,) - if None, use standard all usable pairs.
                     if 'uncensored_only_contribute' -> only pairs where at least one has event=1.
                     We implement two stratified variants:
       c_all      = standard c-index over all pairs (t_i>t_j, event_i=1)
       c_unc_event = c-index using ONLY pairs where BOTH members have event (uncensored).
                     This is the strictest test that predictions are ordered correctly on
                     the *true* event times only, independent of censorship.
       c_half      = pairs where (event_i=1, event_j=0)  ->  one censored, one not.
                     If decomposition gain lives here (gain only on half-censored pairs) it's
                     still valid (not an artifact), but you can decide if you trust it less.
    """
    N = risk.size
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    events = np.asarray(events, dtype=bool).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    # vectorized all pairs via indices to avoid O(N^2) memory for large N:
    # N_test~80 -> 80*79~6400 pairs -> fine O(N^2).
    i, j = np.triu_indices(N, k=1)
    ti, tj = times[i], times[j]
    ei, ej = events[i], events[j]
    # For each pair, decide: which order is comparable, and the "true" ordering?
    # Standard C-index for survival:
    # A comparable pair requires: (t_i != t_j) AND (at least the earlier-time patient is uncensored).
    # Precisely:
    #   if t_i < t_j AND event_i == 1  -> comparable, truth: risk_i > risk_j is concordant
    #   elif t_j < t_i AND event_j == 1 -> comparable, truth: risk_j > risk_i is concordant
    #   else: tie / incomparable -> skip
    #
    # Build a mask of comparables, and "correctly ordered" flag
    cmp_mask = np.zeros(i.shape, dtype=bool)
    truth_smaller_index_has_higher_risk = np.zeros(i.shape, dtype=bool)  # True => want r[i] > r[j]
    # case A: t_i < t_j, ei=1 (earlier uncensored dead, later still alive or dead)
    a = (ti < tj) & ei
    cmp_mask |= a
    truth_smaller_index_has_higher_risk |= a
    # case B: t_j < t_i, ej=1
    b = (tj < ti) & ej
    cmp_mask |= b
    truth_smaller_index_has_higher_risk &= (~b)  # when b holds we want r[j] > r[i], i.e., NOT r[i] > r[j]
    # (for b we'll handle by swapping later, it's easier)
    # For correct concordance handling, write it directly:
    conc = np.zeros(i.shape, dtype=bool)
    disc = np.zeros(i.shape, dtype=bool)
    # Case A
    order = np.sign(risk[i] - risk[j])  # +1 if ri>rj, 0 tie, -1 rj>ri
    rA = a & (order > 0); conc |= rA
    dA = a & (order < 0); disc |= dA
    # Case B: tj<ti, ej=1 -> we want risk_j > risk_i, i.e., sign(risk[j] - risk[i]) > 0
    orderB = np.sign(risk[j] - risk[i])
    rB = b & (orderB > 0); conc |= rB
    dB = b & (orderB < 0); disc |= dB
    # Ties: half credit for comparable pairs that are neither conc nor disc
    ties = cmp_mask & (~conc) & (~disc)
    numerator = conc.astype(np.float64) + 0.5 * ties.astype(np.float64)
    denominator = cmp_mask.astype(np.float64)
    total_c = numerator.sum() / denominator.sum() if denominator.sum() > 0 else np.nan

    # Stratified c-index 1: only pairs with BOTH events=True (two truly-ordered known-event patients)
    both_ei = (ei & ej)
    sub = cmp_mask & both_ei
    num_sub = ((conc + 0.5 * (sub & (~conc) & (~disc))) * sub).sum()
    den_sub = sub.sum()
    c_both_events = num_sub / den_sub if den_sub > 0 else np.nan

    # Stratified 2: only HALF censored pairs (one event one censored), i.e., case A with ej=0 XOR case B with ei=0
    half_censor = cmp_mask & (ei ^ ej)
    num_h = ((conc + 0.5 * (half_censor & (~conc) & (~disc))) * half_censor).sum()
    den_h = half_censor.sum()
    c_half_censor = num_h / den_h if den_h > 0 else np.nan

    return dict(
        c_all=float(total_c),
        c_both_events=float(c_both_events),
        den_both_events=int(den_sub),
        c_half_censor=float(c_half_censor),
        den_half_censor=int(den_h),
        n_pairs_comparable=int(denominator.sum()),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_kind', default='eval_with_censored')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--tau', type=float, default=0.0)
    ap.add_argument('--out_dir', default='outputs/_probe_head_orthog_censor_valid')
    args = ap.parse_args()
    K, lam, tau = args.K, args.lam, args.tau

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    DINO = (ROOT / 'outputs' / 'censored_stage_survival_full_upgrade' /
            'wsi=DINOv2_ViTL_tilefix256_rna=hallmark50_omics' /
            'win0.59-0.65_A0.99_K50' / args.eval_kind / 'phase2')
    RE = re.compile(r"fold_(\d+)_seed(\d+)")
    jobs = []
    for p in sorted(DINO.rglob('summary.json')):
        m = RE.search(str(p))
        if not m: continue
        ckpt = p.parent / 'final.pt'
        if ckpt.is_file():
            jobs.append((int(m.group(1)), int(m.group(2)), ckpt))

    runs = []
    per_patient_records = []
    for (f, s, ckpt) in jobs:
        print(f"\n[RUN] fold={f} seed={s}")
        model, summary = build_model_from_ckpt(ckpt, device=args.device)
        loader, case_table = build_test_loader(summary)
        (risk_orig, H_fused, H_back, times, events, cids,
         W_head_np, b_head) = extract_fused_and_backbone(model, loader, case_table, args.device)
        risk_new, _ = gated_orthog_decompose_risk(
            H_back, risk_orig, W_head_np, b_head, K=K, lam=lam, tau_perp=tau,
        )
        N = risk_orig.size
        censor_rate = 1.0 - events.mean()
        orig_ci = c_index_on_pairs(risk_orig, events, times)
        new_ci  = c_index_on_pairs(risk_new,  events, times)
        run_row = dict(
            fold=f, seed=s, N=N, n_events=int(events.sum()), n_cens=int((~events).sum()),
            censor_rate=float(censor_rate),
            orig_c_all=orig_ci['c_all'], new_c_all=new_ci['c_all'],
            orig_c_both_events=orig_ci['c_both_events'], new_c_both_events=new_ci['c_both_events'],
            orig_c_half=orig_ci['c_half_censor'],   new_c_half=new_ci['c_half_censor'],
            den_both_events=orig_ci['den_both_events'], den_half=orig_ci['den_half_censor'],
        )
        # risk abs differences grouped by event
        dr = np.abs(risk_new - risk_orig)
        run_row.update(dict(
            mean_absdr_all=float(dr.mean()),
            mean_absdr_ev=float(dr[events].mean()) if events.any() else float('nan'),
            mean_absdr_cens=float(dr[~events].mean()) if (~events).any() else float('nan'),
        ))
        # rank shift grouped by event
        rk_o = np.argsort(np.argsort(risk_orig))
        rk_n = np.argsort(np.argsort(risk_new))
        shift = (rk_n - rk_o).astype(int)
        run_row.update(dict(
            mean_abs_shift_all=float(np.abs(shift).mean()),
            mean_abs_shift_ev=float(np.abs(shift[events]).mean()) if events.any() else float('nan'),
            mean_abs_shift_cens=float(np.abs(shift[~events]).mean()) if (~events).any() else float('nan'),
        ))
        for i in range(N):
            per_patient_records.append(dict(
                fold=f, seed=s, idx=i, case_id=str(cids[i]) if cids else f"p{i}",
                event=bool(events[i]), t=float(times[i]),
                r_orig=float(risk_orig[i]), r_new=float(risk_new[i]),
                dr=float(risk_new[i]-risk_orig[i]),
                rank_orig=int(rk_o[i]), rank_new=int(rk_n[i]), shift=int(shift[i]),
            ))
        print(f"  C: orig/all={orig_ci['c_all']:.3f} new/all={new_ci['c_all']:.3f}  "
              f"Δ={(new_ci['c_all']-orig_ci['c_all'])*100:+.2f} pp  "
              f"events={run_row['n_events']} cens={run_row['n_cens']} rate={censor_rate*100:.1f}%")
        print(f"     orig/both_ev={orig_ci['c_both_events']:.3f} new/both_ev={new_ci['c_both_events']:.3f} "
              f"Δ={(new_ci['c_both_events']-orig_ci['c_both_events'])*100:+.2f} pp  "
              f"(den={orig_ci['den_both_events']})")
        print(f"     orig/half={orig_ci['c_half_censor']:.3f} new/half={new_ci['c_half_censor']:.3f} "
              f"Δ={(new_ci['c_half_censor']-orig_ci['c_half_censor'])*100:+.2f} pp")
        print(f"     mean|Δr|: all={run_row['mean_absdr_all']:.3e}  ev={run_row['mean_absdr_ev']:.3e}  "
              f"cens={run_row['mean_absdr_cens']:.3e}")
        runs.append(run_row)

    # Aggregate
    print("\n\n========== AGGREGATE n=%d runs ==========" % len(runs))
    def arr(k): return np.asarray([r[k] for r in runs], dtype=np.float64)
    def pct(tgt, ref): return ((tgt - ref) >= 0).mean() * 100
    orig_c_all = arr('orig_c_all') * 100
    new_c_all  = arr('new_c_all')  * 100
    orig_c_both = arr('orig_c_both_events') * 100
    new_c_both  = arr('new_c_both_events')  * 100
    orig_c_half = arr('orig_c_half') * 100
    new_c_half  = arr('new_c_half')  * 100
    print(f"\n[1] Overall all-pairs C-index:")
    print(f"    baseline  = {orig_c_all.mean():.2f} +/- {orig_c_all.std(ddof=1):.2f} pp")
    print(f"    decomposed= {new_c_all.mean():.2f} +/- {new_c_all.std(ddof=1):.2f} pp")
    d = new_c_all - orig_c_all
    print(f"    Δmean={d.mean():+.2f} pp  Δmedian={np.median(d):+.2f} pp  pct≥={pct(new_c_all,orig_c_all):.0f}%")

    print(f"\n[2] BOTH-EVENTS-only pairs (strictest ground-truth test, censorship-independent):")
    finite = np.isfinite(new_c_both) & np.isfinite(orig_c_both)
    print(f"    baseline  = {np.nanmean(orig_c_both):.2f} +/- {np.nanstd(orig_c_both, ddof=1):.2f} pp  (n_finite={finite.sum()}/{finite.size})")
    print(f"    decomposed= {np.nanmean(new_c_both):.2f} +/- {np.nanstd(new_c_both, ddof=1):.2f} pp")
    d_both = new_c_both - orig_c_both
    print(f"    Δmean={np.nanmean(d_both):+.2f} pp  Δmedian={np.nanmedian(d_both):+.2f} pp  "
          f"pct≥={pct(new_c_both[finite], orig_c_both[finite]):.0f}%")

    print(f"\n[3] HALF-censored pairs (one event, one censored):")
    print(f"    baseline  = {orig_c_half.mean():.2f} +/- {orig_c_half.std(ddof=1):.2f} pp")
    print(f"    decomposed= {new_c_half.mean():.2f} +/- {new_c_half.std(ddof=1):.2f} pp")
    d_half = new_c_half - orig_c_half
    print(f"    Δmean={d_half.mean():+.2f} pp  Δmedian={np.median(d_half):+.2f} pp  "
          f"pct≥={pct(new_c_half, orig_c_half):.0f}%")

    print(f"\n[4] Risk |Δr| by event (are censored patients rewritten more?):")
    m_ev = arr('mean_absdr_ev'); m_cs = arr('mean_absdr_cens')
    print(f"    mean_absdr_ev   = {np.nanmean(m_ev):.3e} +/- {np.nanstd(m_ev, ddof=1):.3e}")
    print(f"    mean_absdr_cens = {np.nanmean(m_cs):.3e} +/- {np.nanstd(m_cs, ddof=1):.3e}")
    print(f"    ratio cens/ev   = {np.nanmean(m_cs / np.maximum(m_ev,1e-20)):.3f}x")

    print(f"\n[5] Rank |Δrank| by event:")
    s_ev = arr('mean_abs_shift_ev'); s_cs = arr('mean_abs_shift_cens')
    print(f"    mean_abs_shift_ev   = {np.nanmean(s_ev):.2f} +/- {np.nanstd(s_ev, ddof=1):.2f}")
    print(f"    mean_abs_shift_cens = {np.nanmean(s_cs):.2f} +/- {np.nanstd(s_cs, ddof=1):.2f}")
    print(f"    ratio cens/ev       = {np.nanmean(s_cs / np.maximum(s_ev,1e-20)):.3f}x")

    print(f"\n[6] Per-fold censor_rate vs Δc_all correlation (should be near 0 for genuine gain):")
    rates = arr('censor_rate')
    # Pair-fold only, average seeds per fold first
    u_folds = sorted(set(r['fold'] for r in runs))
    f_rate, f_d = [], []
    for f in u_folds:
        sub = [r for r in runs if r['fold']==f]
        f_rate.append(np.mean([r['censor_rate'] for r in sub]))
        f_d.append(np.mean([r['new_c_all']-r['orig_c_all'] for r in sub]) * 100)
    if len(f_rate) >= 3:
        corr = np.corrcoef(f_rate, f_d)[0,1]
        print(f"    fold-averaged: rates={[f'{x*100:.1f}%' for x in f_rate]}")
        print(f"                 Δ={[f'{x:+.2f}' for x in f_d]} pp")
        print(f"                 Pearson r = {corr:+.3f}")

    # Save JSON and CSV
    import csv
    (out_dir / 'runs.json').write_text(json.dumps(runs, indent=2))
    csv_path = out_dir / 'runs.csv'
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader(); w.writerows(runs)
    csv2 = out_dir / 'per_patient.csv'
    with csv2.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_patient_records[0].keys()))
        w.writeheader(); w.writerows(per_patient_records)
    print(f"\n  wrote: {csv_path}  {csv2}  runs.json")


if __name__ == '__main__':
    main()
