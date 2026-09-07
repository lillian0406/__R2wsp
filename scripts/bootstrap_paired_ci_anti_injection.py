#!/usr/bin/env python3
"""
Bootstrap paired C-index (Task C):
===================================
For each (fold, seed) we have two saved checkpoints: baseline best.pt and anti_on best.pt.
Load each, predict risk on the SAME test split (the fold's test.csv rows matching case_ids,
respecting anti bank zero-placeholder for non-LUAD subset cases — identical to training logic).

Compute per (fold, seed) ΔCI = CI_anti − CI_base on the same set of patients.
Then B=2000 stratified patient bootstrap (with replacement, keeping anti/base pairings)
to get 95% CI on ΔCI and a paired permutation p-value against H0: ΔCI = 0.

Outputs:
  outputs/sweep_censored_stage_survival_anti_plip_vec_LUAD/bootstrap_ci.json
  outputs/sweep_censored_stage_survival_anti_plip_vec_LUAD/bootstrap_ci.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from r2wsp.data.tcga_dataset import TCGAMultimodalDataset
from r2wsp.models.direct_survival import DirectWSIRNASurvival
from train_censored_stage_survival import build_loader, gather_labels, load_official_split

SPLIT_ROOT = ROOT / "data" / "splits" / "censored_stage_protocol" / "phase1_outer5"
SWEEP = ROOT / "outputs" / "sweep_censored_stage_survival_anti_plip_vec_LUAD"
ANTI_DIR = ROOT / "outputs" / "anti_feature_experiment"


@dataclass
class RunPreds:
    tag: str
    fold: int
    seed: int
    case_ids: list[str]
    times: np.ndarray
    events: np.ndarray   # 1=event, 0=censor
    risk_base: np.ndarray
    risk_anti: np.ndarray

    @property
    def censorship(self) -> np.ndarray:
        return 1.0 - self.events


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def concordance_index(risk: np.ndarray, time: np.ndarray, censorship: np.ndarray) -> float:
    """Exact CI (tie-robust) — matches training's implementation."""
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    censorship = np.asarray(censorship, dtype=np.float64).reshape(-1)
    assert risk.size == time.size == censorship.size
    n = risk.size
    num = 0.0
    den = 0.0
    for i in range(n):
        if float(censorship[i]) == 1.0:
            continue
        for j in range(n):
            if i == j:
                continue
            ti, tj = float(time[i]), float(time[j])
            if ti < tj:
                den += 1.0
                ri, rj = float(risk[i]), float(risk[j])
                if ri > rj:
                    num += 1.0
                elif ri == rj:
                    num += 0.5
    return float(num / den) if den > 0 else float("nan")


def build_dataset_from_split(fold: int, *, use_anti: bool, seed: int,
                             target_col: str = "os_survival_days"):
    import argparse as _argparse
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_censored_stage_survival import (
        read_split_csv,
        resolve_data_paths,
        scan_all_assets,
        build_index,
        apply_cohort_hint,
        dedupe_rows_by_case,
        load_official_split,
        infer_cohort_from_split_dir,
    )
    split_dir = SPLIT_ROOT / f"fold_{fold}"
    class FakeArgs:
        data_root = str(ROOT)
        config = None
        def __getattr__(self, k): return None
    paths = resolve_data_paths(config_path=None, data_root=str(ROOT))
    paths.validate()
    inv = scan_all_assets(paths)
    all_rows = build_index(inv)
    cohort_hint = "LUAD"
    rows = apply_cohort_hint(
        dedupe_rows_by_case(all_rows, source_name="plip_luad_256"),
        cohort_hint=cohort_hint,
        reference_rows=all_rows,
    )
    rows = [r for r in rows if str(r.wsi_feature_source) == "plip_luad_256"]
    case_table, train_case_ids, test_case_ids = load_official_split(split_dir, target_col)
    test_rows = [r for r in rows if str(r.case_id) in test_case_ids]
    cohort_list = ["LUAD"]
    loader = build_loader(
        test_rows,
        seed=int(seed),
        batch_size=16,
        max_tiles=4096,
        rna_mode="vec",
        gene_sets_csv=None,
        gene_id_to_symbol=None,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        use_multi_slide=False,
        multi_slide_tile_budget_mode="per_slide",
        use_anti_features=bool(use_anti),
        anti_feature_dir=str(ANTI_DIR),
        anti_feature_cohorts=cohort_list,
    )
    return loader, None, case_table, test_rows


def load_model_from_run(tag: str, fold: int, seed: int, *, use_anti: bool,
                        tile_dim: int, rna_dim: int, anti_dim: int,
                        hidden_dim: int = 256) -> DirectWSIRNASurvival:
    ckpt_path = SWEEP / tag / f"fold_{fold}_seed_{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DirectWSIRNASurvival(
        tile_dim=tile_dim, rna_dim=rna_dim, hidden_dim=hidden_dim, dropout=0.15,
        model_variant="baseline", aug_dim=5, geo_dim=2, pool_method="attention",
        gate_enabled=False,
        wsi_geo_type="none", wsi_geo_num_points=1, wsi_geo_coord_dim=2,
        wsi_geo_output="points", wsi_geo_position="after_mean", wsi_geo_fusion="concat",
        wsi_b_points_level="case",
        rna_geo_type="none", rna_geo_num_points=1, rna_geo_coord_dim=2,
        rna_geo_output="inherit", rna_geo_position="after_rna_proj", rna_geo_fusion="concat",
        geo_swap=False, use_multi_slide=False, multi_slide_mode="slide_mean_case_attn",
        rna_mode="vec", rna_omic_sizes=None, cross_modal_fusion="concat",
        anti_dim=anti_dim, use_anti_injection=use_anti,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def predict_all(model: DirectWSIRNASurvival, loader, case_table: dict,
                device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns ordered case_ids, times, events (1=death), risk scalar per patient."""
    case_ids: list[str] = []
    times: list[float] = []
    events: list[float] = []
    risks: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels_t, events_t = gather_labels(batch.case_id, case_table, device)
            risk, _ = model(
                tile_tokens=batch.tile_tokens.to(device),
                tile_xy=batch.tile_xy.to(device),
                tile_attn_mask=batch.tile_attn_mask.to(device),
                slide_ids=batch.slide_ids.to(device),
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
                wsi_anti_vec=batch.wsi_anti_vec.to(device) if batch.wsi_anti_vec is not None else None,
                rna_anti_vec=batch.rna_anti_vec.to(device) if batch.rna_anti_vec is not None else None,
            )
            case_ids.extend(batch.case_id)
            times.extend(labels_t.detach().cpu().numpy().tolist())
            events.extend(events_t.detach().cpu().numpy().tolist())
            risks.extend(risk.detach().cpu().reshape(-1).numpy().tolist())
    return case_ids, np.asarray(times), np.asarray(events), np.asarray(risks)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[bootstrap] running on device={device}", flush=True)

    probe_loader, _, _, _ = build_dataset_from_split(0, use_anti=False, seed=0)
    probe_batch = next(iter(probe_loader))
    tile_dim = int(probe_batch.tile_tokens.shape[-1])
    rna_dim = int(probe_batch.rna_vec.shape[-1]) if probe_batch.rna_vec is not None else 1
    anti_dim = 128
    if probe_batch.wsi_anti_vec is not None and probe_batch.rna_anti_vec is not None:
        anti_dim = int(probe_batch.wsi_anti_vec.shape[-1])
    print(f"[bootstrap] tile_dim={tile_dim} rna_dim={rna_dim}", flush=True)

    per_pair: dict[tuple[int, int], RunPreds] = {}
    # Reuse loaders: anti_on loader differs from baseline loader (extra vec keys) but same test rows.
    for fold in range(5):
        for seed in range(3):
            base_loader, _, ct, _ = build_dataset_from_split(fold, use_anti=False, seed=seed)
            anti_loader, _, _, _ = build_dataset_from_split(fold, use_anti=True,  seed=seed)
            model_base = load_model_from_run("baseline", fold, seed, use_anti=False,
                                             tile_dim=tile_dim, rna_dim=rna_dim, anti_dim=anti_dim)
            model_anti = load_model_from_run("anti_on",  fold, seed, use_anti=True,
                                             tile_dim=tile_dim, rna_dim=rna_dim, anti_dim=anti_dim)
            model_base.to(device); model_anti.to(device)
            cids, t, ev, r_base = predict_all(model_base, base_loader, ct, device)
            _, _, _, r_anti = predict_all(model_anti, anti_loader, ct, device)
            per_pair[(fold, seed)] = RunPreds(
                tag="pair", fold=fold, seed=seed, case_ids=cids, times=t, events=ev,
                risk_base=r_base, risk_anti=r_anti,
            )
            ci_base = concordance_index(r_base, t, 1.0 - ev)
            ci_anti = concordance_index(r_anti, t, 1.0 - ev)
            print(f"[pair fold{fold} seed{seed}] N={len(t)} CI_base={ci_base:.4f}  "
                  f"CI_anti={ci_anti:.4f}  Δ={(ci_anti-ci_base)*100:+.2f}pp", flush=True)
            del model_base, model_anti

    # ---------- Bootstrap ----------
    rng = np.random.default_rng(20260730)
    B = 2000
    def delta_macro(per_pair: dict[tuple[int, int], RunPreds]) -> float:
        deltas = []
        for (f, s), p in per_pair.items():
            censor = 1.0 - p.events
            ci_a = concordance_index(p.risk_anti, p.times, censor)
            ci_b = concordance_index(p.risk_base, p.times, censor)
            deltas.append(ci_a - ci_b)
        return float(np.mean(deltas))

    delta_obs = delta_macro(per_pair)
    deltas_boot = np.empty(B, dtype=np.float64)
    keys = list(per_pair.keys())
    # For each bootstrap iteration: within each (fold,seed) resample patients with replacement, keep base/anti aligned.
    for b in range(B):
        boot_pp = {}
        for k in keys:
            p = per_pair[k]
            n = len(p.case_ids)
            idx = rng.integers(0, n, size=n)
            boot_pp[k] = RunPreds(
                tag=p.tag, fold=p.fold, seed=p.seed,
                case_ids=[p.case_ids[i] for i in idx],
                times=p.times[idx], events=p.events[idx],
                risk_base=p.risk_base[idx], risk_anti=p.risk_anti[idx],
            )
        deltas_boot[b] = delta_macro(boot_pp)
    ci_low, ci_high = float(np.quantile(deltas_boot, 0.025)), float(np.quantile(deltas_boot, 0.975))

    # permutation p-value: within each (fold,seed) randomly swap anti/base labels B*2 times? Use B times.
    perm = np.empty(B, dtype=np.float64)
    for b in range(B):
        swap_pp = {}
        for k in keys:
            p = per_pair[k]
            if rng.random() < 0.5:
                swap_pp[k] = p
            else:
                swap_pp[k] = RunPreds(
                    tag=p.tag, fold=p.fold, seed=p.seed, case_ids=p.case_ids,
                    times=p.times, events=p.events,
                    risk_base=p.risk_anti, risk_anti=p.risk_base,  # swap
                )
        perm[b] = delta_macro(swap_pp)
    # two-sided p-value (match t-test logic on |stat|)
    p_perm = float(np.mean(np.abs(perm) >= np.abs(delta_obs)))

    # ---------- Summarize + save ----------
    rows = []
    for (f, s), p in per_pair.items():
        censor = 1.0 - p.events
        ci_b = concordance_index(p.risk_base, p.times, censor)
        ci_a = concordance_index(p.risk_anti, p.times, censor)
        rows.append({"fold": f, "seed": s, "N_test": len(p.case_ids),
                     "n_events": int(p.events.sum()),
                     "pct_censor": f"{(1-p.events.mean())*100:.1f}%",
                     "CI_baseline": f"{ci_b:.6f}",
                     "CI_anti_on": f"{ci_a:.6f}",
                     "delta_pp": f"{(ci_a-ci_b)*100:+.4f}",
                     })
    out_json = SWEEP / "bootstrap_ci.json"
    out_csv = SWEEP / "bootstrap_ci.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "B_bootstrap": B,
        "B_perm": B,
        "N_paired": len(per_pair),
        "delta_observed_pp": delta_obs * 100,
        "delta_boot_mean_pp": float(deltas_boot.mean() * 100),
        "delta_boot_std_pp": float(deltas_boot.std(ddof=1) * 100),
        "delta_95CI_low_pp": ci_low * 100,
        "delta_95CI_high_pp": ci_high * 100,
        "perm_p_value_two_sided": p_perm,
        "rows": rows,
        "elapsed_sec": time.time() - t0,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n======================= BOOTSTRAP RESULTS =======================")
    print(f"  N_paired        : {summary['N_paired']}")
    print(f"  ΔCI observed    : {summary['delta_observed_pp']:+.3f} pp")
    print(f"  ΔCI 95% CI      : [{summary['delta_95CI_low_pp']:+.3f}, {summary['delta_95CI_high_pp']:+.3f}] pp")
    print(f"  ΔCI boot std    : ±{summary['delta_boot_std_pp']:.3f} pp")
    print(f"  permutation p   : {summary['perm_p_value_two_sided']:.4f} (two-sided, swap labels per fold×seed)")
    print(f"  elapsed         : {summary['elapsed_sec']:.1f} s")
    print(f"  bootstrap_ci.json : {out_json}")
    print(f"  bootstrap_ci.csv  : {out_csv}")


if __name__ == "__main__":
    main()
