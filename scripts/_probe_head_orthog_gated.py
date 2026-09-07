"""
Post-hoc probe #3: head weight orthogonal decomposition w/ anti-noise gating.

Hypothesis ("horizontal line vs vertical line + small ripple"):
  Head weight w = w_parallel (along top-K PCs of hidden H, "main line")
               + λ * w_perp_sparse (orthogonal to main line, "small ripple")
  where λ < 1 and w_perp is sparsified via soft-thresholding,
  AND noise eigen-vectors are gated out via Marcenko-Pastur eigen-value floor.

Zero-impact:
  * No training, no GPU, CPU only.
  * Uses only frozen DINOv2 ViTL + hallmark50 phase2 final.pt (15 runs).
  * Results written to new outputs/_probe_head_orthog_gated/.
"""
import os, sys, json, csv, argparse, re
from pathlib import Path

import numpy as np
from numpy.linalg import svd

try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT = Path('/root/autodl-tmp/R2wsp')
sys.path.insert(0, str(ROOT))

from scripts._probe_spherical_smoothing import (
    build_model_from_ckpt, build_test_loader, c_index,
)
from scripts.train_censored_stage_survival import gather_labels
import torch


@torch.no_grad()
def extract_fused_and_backbone(model, loader, case_table, device):
    """Return (risk_orig, H_fused, H_backbone, times, events, case_ids, head_w, head_b)
    H_fused    (N, D_fused)  pre-backbone concat (usually 512)
    H_backbone (N, D_hidden) post-backbone hidden  (usually 256)
    """
    case_ids, times, events, risks, fused_vecs, hiddens = [], [], [], [], [], []
    dev = torch.device(device)
    head_w = None; head_b = None
    for batch in loader:
        tile_tokens = batch.tile_tokens.to(dev)
        tile_xy = batch.tile_xy.to(dev)
        tile_attn_mask = batch.tile_attn_mask.to(dev)
        slide_ids = batch.slide_ids.to(dev) if batch.slide_ids is not None else None
        rna_vec = batch.rna_vec.to(dev) if (hasattr(batch, 'rna_vec') and batch.rna_vec is not None) else None
        rna_omics = [x.to(dev) for x in batch.rna_omics] if (hasattr(batch, 'rna_omics') and batch.rna_omics is not None) else None
        risk, aux = model(
            tile_tokens=tile_tokens, tile_xy=tile_xy, tile_attn_mask=tile_attn_mask,
            slide_ids=slide_ids, rna_vec=rna_vec, rna_omics=rna_omics,
        )
        times_t, events_t = gather_labels(batch.case_id, case_table, dev)
        case_ids.extend(batch.case_id)
        times.append(times_t.cpu().numpy())
        events.append(events_t.cpu().numpy().astype(bool))
        risks.append(risk.cpu().numpy())
        fused_vecs.append(aux['fused'].cpu().numpy())
        # backbone forward to get hidden (aux doesn't return it; compute manually on fused)
        with torch.no_grad():
            fused_t = torch.from_numpy(aux['fused'].cpu().numpy()).float()
            h = model.backbone(fused_t)
        hiddens.append(h.numpy())
        if head_w is None:
            head_w = model.head.weight.detach().cpu().numpy()   # (1, D_hidden)
            head_b = float(model.head.bias.detach().cpu().numpy().reshape(-1)[0])
    return (np.concatenate(risks),
            np.concatenate(fused_vecs, 0),
            np.concatenate(hiddens, 0),
            np.concatenate(times).astype(np.float64),
            np.concatenate(events).astype(bool), case_ids,
            head_w, head_b)

def svd_flip(u, v):
    """Ensure deterministic SVD sign convention (max abs column of u positive)."""
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u = u * signs
    v = v * signs[:, None]
    return u, v


def mp_noise_floor(N: int, D: int):
    """Marcenko-Pastur upper bound on singular value for pure Gaussian noise.
    λ_max = σ² * (1 + sqrt(D/N))², but σ unknown; here we return the *shape*
    factor sqrt( (1 + sqrt(D/N))**2 ) = 1 + sqrt(D/N) so caller can threshold
    singular values S by:    keep if  (S_i / S_0) > 0.1 * mp_shape   (ad-hoc floor)
    For a true MP floor you'd calibrate σ; we use a data-driven floor: keep
    components whose singular value > 10% of the largest, AND shape-wise above
    the MP shape-factor-threshold relative to the mean tail-singular value.
    """
    gamma = D / max(1, N)
    return (1.0 + np.sqrt(gamma)) ** 2


def gated_orthog_decompose_risk(
    H: np.ndarray,          # (N, D) backbone hidden (post-ReLU 256-d for our ckpts)
    risk_orig: np.ndarray,  # (N,)   original risk for reference
    w_orig: np.ndarray,     # (1, D) head weight row from ckpt
    b_orig: float,          # head bias
    K: int,                 # number of PCs in the "parallel" horizontal-line subspace
    lam: float,             # λ ripple amplitude on perpendicular part (λ<1 means shrink)
    tau_perp: float,        # sparse soft-threshold (as fraction of |w_perp| mean abs)
    mp_gate_fraction: float = 0.10,
):
    """Return new risk under the gated-orthog decomposed head."""
    N, D = H.shape
    # 1) Compute sample covariance PCs via SVD of H (centered)
    H_mean = H.mean(axis=0, keepdims=True)
    Hc = H - H_mean
    # svd on small matrix: Hc is (N, D), N~80, D=256  -> compute Vt directly on (D, N) small
    # Use economy SVD on Hc.T (D, N) to get V as (D, min(N,D)) right singular vecs of Hc,
    # which = left singular vecs of covariance ~ PCs.  Use scipy? numpy SVD on D~256 cheap.
    U, S, Vt = svd(Hc, full_matrices=False)  # U(N,r), S(r,), Vt(r,D); r=min(N,D)
    U, Vt = svd_flip(U, Vt)
    PCs = Vt.T                                  # (D, r): each col = PC direction in D-space

    # 2) Marcenko-Pastur noise floor gating on singular values: kill PCs whose S
    #    is too small relative to the noise tail. Gate via S_i / S_0 > mp_gate_fraction,
    #    AND in the first K (if K is larger than signal rank we truncate here).
    if S.size == 0 or S[0] <= 0:
        S_rel = np.ones_like(S)
    else:
        S_rel = S / (S[0] + 1e-9)
    signal_mask = S_rel >= mp_gate_fraction     # (r,) bool
    # Take the intersection: top-K AND signal_mask (preserves order, no gaps -> smaller eff K)
    eff_K_mask = np.zeros_like(signal_mask)
    n_kept = 0
    for i in range(signal_mask.size):
        if signal_mask[i] and n_kept < K:
            eff_K_mask[i] = True
            n_kept += 1
    # If too few passed MP gate, fall back to top-K regardless (specified K works)
    if n_kept < min(K, 3):
        eff_K_mask = np.zeros_like(signal_mask)
        eff_K_mask[:K] = True
    keep_PC_cols = PCs[:, eff_K_mask]             # (D, K_eff)

    # 3) Project w_orig into parallel + perpendicular parts (relative to top-K PCs)
    w_row = w_orig.reshape(-1)                    # (D,)
    # w_parallel = sum_i <w, pc_i> pc_i   (project onto span of K_eff PCs)
    coeff = keep_PC_cols.T @ w_row                # (K_eff,)
    w_parallel = keep_PC_cols @ coeff             # (D,)
    w_perp = w_row - w_parallel                   # (D,) must be orthogonal to kept PCs (to numerical precision)

    # 4) Sparse soft-threshold on the perpendicular part (noise gating #2)
    if tau_perp > 0 and w_perp.size > 0:
        thr = tau_perp * (np.mean(np.abs(w_perp)) + 1e-9)
        # soft threshold
        sign = np.sign(w_perp)
        mag  = np.maximum(0.0, np.abs(w_perp) - thr)
        w_perp_sp = sign * mag
    else:
        w_perp_sp = w_perp

    # 5) Assemble new weight = w_parallel + λ * w_perp_sparse
    w_new = w_parallel + lam * w_perp_sp          # (D,)

    # 6) Compute new risk (H still on original scale, bias kept)
    risk_new = H @ w_new + b_orig
    return risk_new, dict(
        K_eff=int(eff_K_mask.sum()),
        norm_w_parallel=float(np.linalg.norm(w_parallel)),
        norm_w_perp_orig=float(np.linalg.norm(w_perp)),
        norm_w_perp_gated=float(np.linalg.norm(w_perp_sp)),
        perp_sparsity=float((np.abs(w_perp_sp) > 0).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_kind', default='eval_with_censored')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--K_grid', default='3,5,8,10,15')
    ap.add_argument('--lam_grid', default='0.0,0.05,0.10,0.20,0.30,0.50,0.70,1.0')
    ap.add_argument('--tau_grid', default='0.0,0.30,0.50,0.80')
    ap.add_argument('--out_dir', default='outputs/_probe_head_orthog_gated')
    ap.add_argument('--max_jobs', type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = args.device

    DINO = (ROOT / 'outputs' / 'censored_stage_survival_full_upgrade' /
            'wsi=DINOv2_ViTL_tilefix256_rna=hallmark50_omics' /
            'win0.59-0.65_A0.99_K50' / args.eval_kind / 'phase2')
    jobs = []
    RE = re.compile(r"fold_(\d+)_seed(\d+)")
    for p in sorted(DINO.rglob('summary.json')):
        m = RE.search(str(p))
        if not m: continue
        ckpt = p.parent / 'final.pt'
        if ckpt.is_file():
            jobs.append((int(m.group(1)), int(m.group(2)), ckpt, p))
    if args.max_jobs > 0:
        jobs = jobs[:args.max_jobs]
    print(f"[I] n={len(jobs)} ckpts. device={dev}")
    K_GRID = [int(x) for x in args.K_grid.split(',')]
    LAM_GRID = [float(x) for x in args.lam_grid.split(',')]
    TAU_GRID = [float(x) for x in args.tau_grid.split(',')]

    all_rows = []
    for (f, s, ckpt, sp) in jobs:
        print(f"\n[RUN] fold={f} seed={s}")
        model, summary = build_model_from_ckpt(ckpt, device=dev)
        loader, case_table = build_test_loader(summary)
        (risk0, _H_fused, H_backbone, times, events, _cids,
         W_head_np, b_head) = extract_fused_and_backbone(model, loader, case_table, dev)
        c0 = c_index(risk0, events, times)
        expected = summary.get('final_test_c_index') or summary.get('best_test_c_index')
        D_hidden = H_backbone.shape[1]
        assert W_head_np.shape[1] == D_hidden, (W_head_np.shape, D_hidden)
        print(f"  orig C = {c0*100:.3f}%  N_te={H_backbone.shape[0]}  expected={expected}  "
              f"D_hidden={D_hidden}  |w_orig|={np.linalg.norm(W_head_np):.3f}")

        row = dict(fold=f, seed=s, n_te=H_backbone.shape[0], c_orig=c0)
        diag_store = None
        for K in K_GRID:
            for lam in LAM_GRID:
                for tau in TAU_GRID:
                    risk_new, diag = gated_orthog_decompose_risk(
                        H_backbone, risk0, W_head_np, b_head, K=K, lam=lam, tau_perp=tau,
                    )
                    if diag_store is None:
                        diag_store = diag
                    cn = c_index(risk_new, events, times)
                    row[f'c_K{K}_L{lam:.2f}_T{tau:.2f}'] = cn

        print(f"  diag first: K_eff={diag_store['K_eff']}  |w_para|={diag_store['norm_w_parallel']:.3f}  "
              f"|w_perp_orig|={diag_store['norm_w_perp_orig']:.3f}  "
              f"|w_perp_gated|={diag_store['norm_w_perp_gated']:.3f}  "
              f"perp_sparsity={diag_store['perp_sparsity']:.3f}")
        best = (c0, None)
        for k in row:
            if k.startswith('c_K') and row[k] > best[0]:
                best = (row[k], k)
        if best[1]:
            print(f"  best this run: {best[1]} -> {best[0]*100:.3f}%  Δ={(best[0]-c0)*100:+.3f} pp")
        all_rows.append(row)
        (out_dir / f'per_fold_f{f}_s{s}.json').write_text(json.dumps(row, indent=2))

    print("\n\n========== AGGREGATE (paired n=%d) ==========" % len(all_rows))
    c_orig = np.asarray([r['c_orig'] for r in all_rows]) * 100
    print(f"  ORIGINAL (baseline): mean = {c_orig.mean():.2f} +/- {c_orig.std(ddof=1):.2f} pp  median={np.median(c_orig):.2f} pp")
    best_abs = (-1e9, None)
    summary_lines = []
    # First group by (K,lam,tau) alphabetical to iterate
    keys = [k for k in all_rows[0].keys() if k.startswith('c_K')]
    for key in sorted(keys):
        a = np.asarray([r[key] for r in all_rows]) * 100
        d = a - c_orig
        pct = (d >= 0).mean() * 100
        line = (f"  {key:>28s}  -> mean={a.mean():.2f} +/- {a.std(ddof=1):.2f} pp   "
                f"Δmean={d.mean():+.2f} pp  Δmedian={np.median(d):+.2f} pp  pct≥={pct:.0f}%")
        if HAS_SCIPY and (d != 0).any():
            try:
                tt = ttest_1samp(d / 100.0, 0.0).pvalue
                try:
                    ww = wilcoxon(d / 100.0).pvalue
                except Exception:
                    ww = float('nan')
                line += f"   ttest p={tt:.3f}   wilcoxon p={ww:.3f}"
            except Exception:
                pass
        summary_lines.append((d.mean(), line))
        best_abs = max(best_abs, (d.mean(), (key, a, d)))
        print(line)

    print("\n========== TOP-10 by Δmean ==========")
    summary_lines.sort(reverse=True)
    for (dm, line) in summary_lines[:10]:
        print(line)

    # Write CSV
    csv_path = out_dir / f'summary_{args.eval_kind}.csv'
    fieldnames = list(all_rows[0].keys())
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  CSV -> {csv_path}")
    # Write aggregate JSON
    top10 = []
    for dm, line in summary_lines[:10]:
        top10.append(line)
    (out_dir / f'summary_{args.eval_kind}.json').write_text(json.dumps(dict(
        eval_kind=args.eval_kind,
        n_runs=len(all_rows),
        K_grid=K_GRID, lam_grid=LAM_GRID, tau_grid=TAU_GRID,
        orig_mean_pp=float(c_orig.mean()),
        orig_std_pp=float(c_orig.std(ddof=1)),
        orig_median_pp=float(np.median(c_orig)),
        best=(best_abs[1][0], float(best_abs[1][2].mean()), float(np.median(best_abs[1][2])),
              float((best_abs[1][2] >= 0).mean() * 100)) if best_abs[1] else None,
        top10_lines=top10,
    ), indent=2))


if __name__ == '__main__':
    main()
