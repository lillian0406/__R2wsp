#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import train_censored_stage_survival as train_mod
from r2wsp.models.direct_survival import DirectWSIRNASurvival


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _tpm_case_coverage(rna_path: str | None) -> set[str] | None:
    if rna_path is None:
        return None
    p = Path(str(rna_path)).resolve()
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    cov: set[str] = set()
    for col in cols:
        if col.startswith("TCGA-"):
            cov.add(col)
            cov.add(col[:12])
    return cov


def export_case_risk(
    *,
    run_dir: Path,
    ckpt_path: Path,
    export_tag: str | None,
    split_dir_override: Path | None,
    target_col_override: str | None,
    splits: list[str],
    export_batch_size: int | None,
    device_override: str | None,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "args" not in ckpt:
        raise ValueError(f"checkpoint does not contain args: {ckpt_path}")
    args_ns = argparse.Namespace(**ckpt["args"])
    if device_override is not None:
        setattr(args_ns, "device", device_override)

    device = torch.device(str(getattr(args_ns, "device", "cpu")))
    paths = train_mod.resolve_data_paths(config_path=getattr(args_ns, "config", None), data_root=getattr(args_ns, "data_root", None))
    paths.validate()

    dataset_rna_mode = "omics" if str(getattr(args_ns, "rna_mode", "vec")) in {"omics", "omics_attn"} else "vec"
    gene_sets_csv = None
    gene_id_to_symbol = None
    if dataset_rna_mode == "omics":
        gene_sets_csv = train_mod.resolve_gene_sets_csv(paths.raw_rna_root, getattr(args_ns, "rna_gene_sets_csv", None))
        gtf_path = train_mod.resolve_gene_annotation_gtf(getattr(args_ns, "gene_annotation_gtf", None))
        if gtf_path is not None:
            gene_id_to_symbol = train_mod.load_gene_id_to_symbol_map(gtf_path)

    inv = train_mod.scan_all_assets(paths)
    all_rows = train_mod.build_index(inv)
    split_dir = (split_dir_override.resolve() if split_dir_override is not None else Path(str(getattr(args_ns, "split_dir"))).resolve())
    cohort_hint = str(getattr(args_ns, "cohort_hint", "")) or train_mod.infer_cohort_from_split_dir(split_dir)
    base_rows = all_rows if bool(getattr(args_ns, "use_multi_slide", False)) else train_mod.dedupe_rows_by_case(all_rows, source_name=str(getattr(args_ns, "wsi_feature_source")))
    rows = train_mod.apply_cohort_hint(base_rows, cohort_hint=cohort_hint, reference_rows=all_rows)
    rows = [r for r in rows if str(r.wsi_feature_source) == str(getattr(args_ns, "wsi_feature_source"))]

    tpm_cov: set[str] | None = None
    if dataset_rna_mode in {"vec", "omics"} and str(getattr(args_ns, "rna_mode", "vec")) in {"vec", "omics", "omics_attn"}:
        any_rna_path = next((r.rna_path for r in rows if getattr(r, "rna_path", None) is not None), None)
        tpm_cov = _tpm_case_coverage(None if any_rna_path is None else str(any_rna_path))

    target_col = str(target_col_override) if target_col_override is not None else str(getattr(args_ns, "target_col", "dss_survival_days"))
    case_table, train_case_ids, test_case_ids = train_mod.load_official_split(split_dir, target_col)
    train_case_ids, val_case_ids = train_mod.split_train_val_case_ids(
        sorted(train_case_ids),
        int(getattr(args_ns, "seed", 0)),
        float(getattr(args_ns, "val_frac", 0.2)),
        case_table,
        mode=str(getattr(args_ns, "val_split_mode", "survival_stratified")),
        n_time_bins=int(getattr(args_ns, "val_time_bins", 4)),
    )
    split_case_ids = {"train": set(train_case_ids), "val": set(val_case_ids), "test": set(test_case_ids)}
    if tpm_cov is not None:
        split_case_ids = {k: {cid for cid in v if (cid in tpm_cov)} for k, v in split_case_ids.items()}
    split_rows = {k: [r for r in rows if (r.case_id is not None and str(r.case_id) in split_case_ids[k])] for k in split_case_ids}

    batch_size = int(export_batch_size if export_batch_size is not None else getattr(args_ns, "batch_size", 8))
    max_tiles_eval = getattr(args_ns, "max_tiles_eval", None)
    if max_tiles_eval is None:
        max_tiles_eval = getattr(args_ns, "max_tiles", None)
    max_tiles_eval = None if max_tiles_eval in (None, "None") else int(max_tiles_eval)

    tile_sampling_eval_raw = getattr(args_ns, "tile_sampling_eval", None)
    if tile_sampling_eval_raw in (None, "None", ""):
        tile_sampling_eval_raw = getattr(args_ns, "tile_sampling", "prefix")
    tile_sampling_eval = str(tile_sampling_eval_raw)
    tile_sampling_seed = int(getattr(args_ns, "tile_sampling_seed", 0))

    loaders = {
        split: train_mod.build_loader(
            split_rows[split],
            seed=int(getattr(args_ns, "seed", 0)),
            batch_size=batch_size,
            max_tiles=max_tiles_eval,
            tile_sampling=tile_sampling_eval,
            tile_sampling_seed=tile_sampling_seed,
            rna_mode=dataset_rna_mode,
            gene_sets_csv=gene_sets_csv,
            gene_id_to_symbol=gene_id_to_symbol,
            shuffle=False,
            num_workers=int(getattr(args_ns, "num_workers", 0)),
            pin_memory=bool(getattr(args_ns, "pin_memory", False)),
            use_multi_slide=bool(getattr(args_ns, "use_multi_slide", False)),
            multi_slide_tile_budget_mode=str(getattr(args_ns, "multi_slide_tile_budget_mode", "per_slide")),
            use_anti_features=bool(getattr(args_ns, "use_anti_injection", False)),
            anti_feature_dir=str(getattr(args_ns, "anti_feature_dir", "outputs/anti_feature_experiment")),
            anti_feature_cohorts=str(getattr(args_ns, "anti_feature_cohorts", "")) or None,
            require_both_modalities=str(getattr(args_ns, "require_both_modalities", "true")),
        )
        for split in splits
    }

    sample_loader = next((ldr for ldr in loaders.values() if len(ldr.dataset) > 0), None)
    if sample_loader is None:
        raise ValueError("empty loaders: no cases to export")
    sample_batch = next(iter(sample_loader))
    tile_dim = int(sample_batch.tile_tokens.shape[-1])
    rna_dim = int(sample_batch.rna_vec.shape[-1]) if sample_batch.rna_vec is not None else 1
    rna_omic_sizes = [int(x.shape[-1]) for x in sample_batch.rna_omics] if sample_batch.rna_omics is not None else None
    anti_dim = 128
    if sample_batch.wsi_anti_vec is not None and sample_batch.rna_anti_vec is not None:
        anti_dim = int(sample_batch.wsi_anti_vec.shape[-1])

    model_variant = str(getattr(args_ns, "model_variant", "baseline"))
    fusion_mode = str(getattr(args_ns, "fusion_mode", "concat"))
    if model_variant == "baseline" and fusion_mode == "gated":
        model_variant = "gate_only"
    gate_enabled: bool | None = None
    gate_enabled_arg = str(getattr(args_ns, "gate_enabled", "auto"))
    if gate_enabled_arg == "true":
        gate_enabled = True
    elif gate_enabled_arg == "false":
        gate_enabled = False

    model = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=int(getattr(args_ns, "hidden_dim", 256)),
        dropout=float(getattr(args_ns, "dropout", 0.0)),
        model_variant=model_variant,
        aug_dim=int(getattr(args_ns, "aug_dim", 64)),
        geo_dim=int(getattr(args_ns, "geo_dim", 64)),
        pool_method=str(getattr(args_ns, "pool_method", "attention")),
        gate_enabled=gate_enabled,
        wsi_geo_type=str(getattr(args_ns, "wsi_geo_type", "none")),
        wsi_geo_num_points=int(getattr(args_ns, "wsi_geo_num_points", 24)),
        wsi_geo_coord_dim=int(getattr(args_ns, "wsi_geo_coord_dim", 3)),
        wsi_geo_output=str(getattr(args_ns, "wsi_geo_output", "points")),
        wsi_geo_position=str(getattr(args_ns, "wsi_geo_position", "after_mean")),
        wsi_geo_fusion=str(getattr(args_ns, "wsi_geo_fusion", "concat")),
        wsi_b_points_level=str(getattr(args_ns, "wsi_b_points_level", "case")),
        rna_geo_type=str(getattr(args_ns, "rna_geo_type", "none")),
        rna_geo_num_points=int(getattr(args_ns, "rna_geo_num_points", 6)),
        rna_geo_coord_dim=int(getattr(args_ns, "rna_geo_coord_dim", 3)),
        rna_geo_output=str(getattr(args_ns, "rna_geo_output", "inherit")),
        rna_geo_position=str(getattr(args_ns, "rna_geo_position", "after_rna_proj")),
        rna_geo_fusion=str(getattr(args_ns, "rna_geo_fusion", "concat")),
        geo_swap=bool(getattr(args_ns, "geo_swap", False)),
        use_multi_slide=bool(getattr(args_ns, "use_multi_slide", False)),
        multi_slide_mode=str(getattr(args_ns, "multi_slide_mode", "slide_mean_case_attn")),
        rna_mode=str(getattr(args_ns, "rna_mode", "vec")),
        rna_omic_sizes=rna_omic_sizes,
        cross_modal_fusion=str(getattr(args_ns, "cross_modal_fusion", "concat")),
        anti_dim=int(anti_dim),
        use_anti_injection=bool(getattr(args_ns, "use_anti_injection", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    out_tag = str(export_tag) if export_tag is not None else ckpt_path.stem
    out_root = run_dir / "case_risk_exports" / out_tag
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "export_tag": out_tag,
        "split_dir": str(split_dir),
        "splits": splits,
        "device": str(device),
        "export_batch_size": batch_size,
        "max_tiles_eval": None if max_tiles_eval is None else int(max_tiles_eval),
        "tile_sampling_eval": tile_sampling_eval,
        "tile_sampling_seed": tile_sampling_seed,
        "target_col": target_col,
    }

    for split in splits:
        loader = loaders[split]
        rows_out: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:
                times, events = train_mod.gather_labels(batch.case_id, case_table, device)
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
                risk = risk.detach().cpu().reshape(-1).tolist()
                times_cpu = times.detach().cpu().reshape(-1).tolist()
                events_cpu = events.detach().cpu().reshape(-1).tolist()
                for i, case_id in enumerate(batch.case_id):
                    e = float(events_cpu[i])
                    rows_out.append(
                        {
                            "case_id": str(case_id),
                            "risk": float(risk[i]),
                            "event_time": float(times_cpu[i]),
                            "event": e,
                            "censorship": float(1.0 - e),
                        }
                    )
        _write_csv(out_root / f"{split}_cases.csv", rows_out, ["case_id", "risk", "event_time", "event", "censorship"])
        manifest[f"{split}_n"] = len(rows_out)

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Directory that contains summary.json / histories/ etc (used only as an export anchor).")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--export_tag", default=None, help="Optional subfolder name under <run_dir>/case_risk_exports/ (default: checkpoint stem)")
    ap.add_argument("--split_dir_override", default=None, help="Optional split_dir override for exporting on a different case universe")
    ap.add_argument("--target_col_override", default=None)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--export_batch_size", type=int, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    args = ap.parse_args()

    splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    valid = {"train", "val", "test"}
    if not splits or any(x not in valid for x in splits):
        raise ValueError("splits must be a comma-separated subset of train,val,test")

    export_case_risk(
        run_dir=Path(args.run_dir).resolve(),
        ckpt_path=Path(args.ckpt_path).resolve(),
        export_tag=None if args.export_tag in (None, "None", "") else str(args.export_tag),
        split_dir_override=None if args.split_dir_override in (None, "None", "") else Path(str(args.split_dir_override)),
        target_col_override=None if args.target_col_override in (None, "None", "") else str(args.target_col_override),
        splits=splits,
        export_batch_size=args.export_batch_size,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
