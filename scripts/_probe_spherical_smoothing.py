import os, sys, json, csv, argparse, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT = Path('/root/autodl-tmp/R2wsp')
sys.path.insert(0, str(ROOT))

from scripts.train_censored_stage_survival import (
    build_loader, load_official_split, apply_cohort_hint, dedupe_rows_by_case,
    load_gene_id_to_symbol_map, gather_labels,
)
from r2wsp.data import build_index, scan_all_assets
from r2wsp.data.build_index import IndexRow
from src.r2wsp.models.direct_survival import DirectWSIRNASurvival


def c_index(risk, event, time):
    risk = np.asarray(risk, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=np.float64)
    n = risk.shape[0]
    num = 0.0; den = 0.0
    for i in range(n):
        if not event[i]: continue
        for j in range(n):
            if i == j: continue
            if time[i] < time[j]:
                den += 1.0
                if risk[i] > risk[j]: num += 1.0
                elif risk[i] == risk[j]: num += 0.5
    return num / den if den > 0 else float('nan')


def build_model_from_ckpt(ckpt_path, device='cpu'):
    sd_all = torch.load(ckpt_path, map_location='cpu')
    sd = sd_all['model'] if 'model' in sd_all else (sd_all.get('state_dict') or sd_all)
    summary_path = Path(ckpt_path).parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    # Infer dimensions from state_dict keys (consistent with train script)
    tile_dim = int(sd['wsi_proj.0.weight'].shape[1])   # 1024 for DINOv2 ViTL
    hidden_dim = int(summary['hidden_dim'])
    rna_omic_sizes = []
    for i in range(200):
        k = f'rna_omics_encoder.sig_networks.{i}.0.weight'
        if k in sd:
            rna_omic_sizes.append(int(sd[k].shape[1]))
        else:
            break
    rna_dim = sum(rna_omic_sizes)
    if 'rna_proj.0.weight' in sd:
        rna_dim_from_proj = int(sd['rna_proj.0.weight'].shape[1])
        if rna_dim_from_proj > 0 and rna_dim_from_proj != rna_dim:
            rna_dim = rna_dim_from_proj
            rna_omic_sizes = [rna_dim] if not rna_omic_sizes else rna_omic_sizes
    aug_dim = int(sd['aug_proj.weight'].shape[1]) - 2*hidden_dim if 'aug_proj.weight' in sd else 0
    geo_dim = int(sd['geo_proj.weight'].shape[1]) - 2 if 'geo_proj.weight' in sd else 0
    gate_enabled = 'gate_net.weight' in sd
    wsi_geo_type = summary.get('wsi_geo_type', 'none')
    rna_geo_type = summary.get('rna_geo_type', 'none')
    has_wsi_geo = str(wsi_geo_type).lower() not in ('none','') and str(wsi_geo_type) != 'none'
    has_rna_geo = str(rna_geo_type).lower() not in ('none','') and str(rna_geo_type) != 'none'
    anti_dim = (
        int(sd['anti_injection.wsi_proj.0.weight'].shape[0])
        if 'anti_injection.wsi_proj.0.weight' in sd else 128
    )
    kwargs = dict(
        tile_dim=tile_dim, rna_dim=rna_dim, hidden_dim=hidden_dim,
        dropout=float(summary['dropout']), model_variant=str(summary.get('model_variant','baseline')),
        aug_dim=aug_dim, geo_dim=geo_dim,
        pool_method=str(summary.get('pool_method','attention')),
        gate_enabled=gate_enabled,
        wsi_geo_type=str(wsi_geo_type), wsi_geo_num_points=16, wsi_geo_coord_dim=2,
        wsi_geo_output=('points' if has_wsi_geo else 'points'),
        wsi_geo_position='after_mean',
        wsi_geo_fusion=str(summary.get('wsi_geo_fusion','concat')),
        wsi_b_points_level='slide',
        rna_geo_type=str(rna_geo_type), rna_geo_num_points=16, rna_geo_coord_dim=2,
        rna_geo_output=('points' if has_rna_geo else 'points'),
        rna_geo_position='after_rna_proj',
        rna_geo_fusion=str(summary.get('rna_geo_fusion','concat')),
        geo_swap=bool(summary.get('geo_swap', False)),
        use_multi_slide=bool(summary.get('use_multi_slide', False)),
        multi_slide_mode=str(summary.get('multi_slide_mode','slide_mean_case_attn')),
        rna_mode=str(summary.get('rna_mode','omics')),
        rna_omic_sizes=rna_omic_sizes if len(rna_omic_sizes) else None,
        cross_modal_fusion=str(summary.get('cross_modal_fusion','concat')),
        anti_dim=anti_dim,
        use_anti_injection=bool(summary.get('use_anti_injection', False)),
    )
    model = DirectWSIRNASurvival(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), summary


def build_test_loader(summary):
    from r2wsp.data import resolve_data_paths
    cohort_hint = summary.get('cohort_hint') or 'LUAD'
    target_col = summary['target_col']
    wsi_feature_source = summary['wsi_feature_source']
    rna_mode = summary['rna_mode']
    split_dir = Path(summary['split_dir'])
    max_tiles = int(summary['max_tiles'])
    multi_slide_tile_budget_mode = summary.get('multi_slide_tile_budget_mode', 'per_slide')
    gene_gtf = summary.get('gene_annotation_gtf')
    gene_sets_csv = summary.get('rna_gene_sets_csv')
    gene_id_to_symbol = None
    if gene_gtf:
        gene_gtf_path = Path(str(gene_gtf))
        if gene_gtf_path.is_file():
            gene_id_to_symbol = load_gene_id_to_symbol_map(gene_gtf_path)
    data_root = ROOT / 'data'
    cfg_yaml = ROOT / 'configs' / 'data_paths.yaml'
    paths = resolve_data_paths(config_path=str(cfg_yaml), data_root=str(data_root))
    inv = scan_all_assets(paths)
    all_rows = build_index(inv)
    rows = apply_cohort_hint(
        all_rows if bool(summary.get('use_multi_slide')) else dedupe_rows_by_case(all_rows, source_name=str(wsi_feature_source)),
        cohort_hint=cohort_hint, reference_rows=all_rows,
    )
    rows = [r for r in rows if str(r.wsi_feature_source) == str(wsi_feature_source)]
    case_table, _, test_case_ids = load_official_split(split_dir, str(target_col))
    test_rows = [r for r in rows if str(r.case_id) in test_case_ids]
    loader = build_loader(
        test_rows,
        seed=0,
        batch_size=int(summary['batch_size']), shuffle=False,
        max_tiles=max_tiles, num_workers=0, pin_memory=False,
        rna_mode=rna_mode,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        use_multi_slide=bool(summary.get('use_multi_slide', False)),
        multi_slide_tile_budget_mode=multi_slide_tile_budget_mode,
        use_anti_features=bool(summary.get('use_anti_injection', False)),
        anti_feature_dir=None,
        anti_feature_cohorts=None,
    )
    return loader, case_table


@torch.no_grad()
def extract_outputs(model, loader, case_table, device):
    case_ids, times, events, risks, fused_vecs = [], [], [], [], []
    dev = torch.device(device)
    for batch in loader:
        tile_tokens = batch.tile_tokens.to(dev)
        tile_xy = batch.tile_xy.to(dev)
        tile_attn_mask = batch.tile_attn_mask.to(dev)
        slide_ids = batch.slide_ids.to(dev) if batch.slide_ids is not None else None
        rna_vec = batch.rna_vec.to(dev) if (hasattr(batch,'rna_vec') and batch.rna_vec is not None) else None
        rna_omics = [x.to(dev) for x in batch.rna_omics] if (hasattr(batch,'rna_omics') and batch.rna_omics is not None) else None
        risk, aux = model(
            tile_tokens=tile_tokens, tile_xy=tile_xy, tile_attn_mask=tile_attn_mask,
            slide_ids=slide_ids, rna_vec=rna_vec, rna_omics=rna_omics,
        )
        times_t, events_t = gather_labels(batch.case_id, case_table, dev)
        case_ids.extend(batch.case_id)
        times.append(times_t.cpu().numpy())
        events.append(events_t.cpu().numpy().astype(bool))
        risks.append(risk.cpu().numpy())
        fused_vecs.append(aux['fused'].cpu().numpy())   # (B, D_fused) - this feeds backbone directly
    return (np.concatenate(risks), np.concatenate(fused_vecs, 0),
            np.concatenate(times).astype(np.float64),
            np.concatenate(events).astype(bool), case_ids)


def spherical_smooth(H, k=15, n_iter=5, temperature=0.05, eps=1e-4, self_ring_min=0.1):
    N, D = H.shape
    mu = F.normalize(torch.from_numpy(H.astype(np.float32)), dim=-1, eps=eps)
    if N <= 2 or k <= 1:
        return mu.numpy()
    with torch.no_grad():
        sim = mu @ mu.T
        k_eff = max(2, min(k, N-1))
        topk_v, topk_idx = torch.topk(sim, k_eff+1, dim=-1)
        topk_idx = topk_idx[:, 1:]
        topk_v = topk_v[:, 1:]
    logits = (topk_v - topk_v.amax(dim=-1, keepdim=True)) / temperature
    W = F.softmax(logits, dim=-1)
    self_w = 1.0 - W.sum(dim=-1)
    self_w = torch.clamp(self_w, min=self_ring_min, max=1.0)
    W = W * (1.0 - self_w).unsqueeze(-1) / (W.sum(dim=-1, keepdim=True) + eps)
    for it in range(n_iter):
        idx = topk_idx.reshape(-1)
        neighbors = mu[idx].view(N, k_eff, D)
        dot = (neighbors * mu.unsqueeze(1)).sum(-1).clamp(-1+eps, 1-eps)
        theta = torch.acos(dot)
        sin_t = theta.sin().clamp(min=eps)
        scale = theta / sin_t
        tang = scale.unsqueeze(-1) * (neighbors - dot.unsqueeze(-1) * mu.unsqueeze(1))
        tang_avg = (tang * W.unsqueeze(-1)).sum(1)
        t_norm = tang_avg.norm(dim=-1, keepdim=True).clamp(min=eps)
        mu_new = torch.cos(t_norm) * mu + torch.sin(t_norm) * (tang_avg / t_norm)
        mix = self_w.unsqueeze(-1) * mu + (1-self_w).unsqueeze(-1) * mu_new
        mu = F.normalize(mix, dim=-1, eps=eps)
    return mu.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_kind', default='eval_with_censored')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--k_grid', default='5,10,20,40,80')
    ap.add_argument('--temp_grid', default='0.02,0.05,0.10')
    ap.add_argument('--out_dir', default='outputs/_probe_spherical_smoothing')
    ap.add_argument('--max_jobs', type=int, default=0, help='0 = all 15 runs')
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
    K_GRID = [int(x) for x in args.k_grid.split(',')]
    T_GRID = [float(x) for x in args.temp_grid.split(',')]
    results = []
    for (f, s, ckpt, sp) in jobs:
        print(f"\n[RUN] fold={f} seed={s}")
        model, summary = build_model_from_ckpt(ckpt, device=dev)
        loader, case_table = build_test_loader(summary)
        risk0, H0, times, events, cids = extract_outputs(model, loader, case_table, dev)
        c0 = c_index(risk0, events, times)
        expected = summary.get('final_test_c_index') or summary.get('best_test_c_index')
        print(f"    orig C = {c0*100:.3f}%  N_te={risk0.size}  expected={expected}")
        row = dict(fold=f, seed=s, n_te=risk0.size, c_orig=c0)
        # Smooth FUSED (pre-backbone), then re-compute backbone+head on smoothed fused.
        # fused_vecs: (N_te, D_fused)  D_fused=512 for concat DINOv2+hallmark50.
        norms = np.linalg.norm(H0, axis=-1, keepdims=True) + 1e-8
        for k in K_GRID:
            for t in T_GRID:
                Hsu = spherical_smooth(H0, k=k, temperature=t)
                Hs = torch.from_numpy(Hsu * norms).to(torch.float32).to(dev)
                with torch.no_grad():
                    hidden_s = model.backbone(Hs)
                    risk_s = model.head(hidden_s).squeeze(-1).cpu().numpy()
                cn = c_index(risk_s, events, times)
                key = f'k{k}_t{t:.2f}'
                row[f'c_s_{key}'] = cn
                print(f"      k={k:>3d} T={t:.2f}  C={cn*100:.3f}%  Δ={(cn-c0)*100:+.3f} pp")
        results.append(row)
        (out_dir / f'per_fold_f{f}_s{s}.json').write_text(json.dumps(row, indent=2))

    print("\n========== AGGREGATE (paired n=%d) ==========" % len(results))
    c_orig = np.asarray([r['c_orig'] for r in results]) * 100
    print(f"  ORIGINAL hidden: mean = {c_orig.mean():.2f} +/- {c_orig.std(ddof=1):.2f} pp  median={np.median(c_orig):.2f} pp")
    best_abs = (-1e9, None)
    for k in K_GRID:
        for t in T_GRID:
            key = f'c_s_k{k}_t{t:.2f}'
            a = np.asarray([r[key] for r in results]) * 100
            d = a - c_orig
            pct = (d>=0).mean()*100
            line = (f"  k={k:>3d} T={t:.2f}  -> mean={a.mean():.2f} +/- {a.std(ddof=1):.2f} pp   "
                    f"Δmean={d.mean():+.2f} pp  Δmedian={np.median(d):+.2f} pp  pct≥={pct:.0f}%")
            if HAS_SCIPY and (d!=0).any():
                try:
                    tt = ttest_1samp(d/100, 0.0).pvalue
                    ww = wilcoxon(d/100).pvalue
                    line += f"   ttest p={tt:.3f}   wilcoxon p={ww:.3f}"
                except Exception:
                    pass
            print(line)
            if d.mean() > best_abs[0]:
                best_abs = (d.mean(), (k,t,a,d))
    csv_path = out_dir / f'summary_{args.eval_kind}.csv'
    keys = list(results[0].keys())
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(results)
    print(f"\n  CSV -> {csv_path}")
    summary_out = dict(eval_kind=args.eval_kind, n_runs=len(results),
                       k_grid=K_GRID, temp_grid=T_GRID,
                       orig_mean_pp=float(c_orig.mean()), orig_std_pp=float(c_orig.std(ddof=1)),
                       best=(best_abs[1][0], best_abs[1][1], float(best_abs[1][3].mean()),
                             float(np.median(best_abs[1][3])), float((best_abs[1][3]>=0).mean())) if best_abs[1] else None,
                       runs=results)
    (out_dir / f'summary_{args.eval_kind}.json').write_text(json.dumps(summary_out, indent=2))


if __name__ == '__main__':
    main()
