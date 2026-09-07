from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for _path in (str(_SCRIPTS), str(_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import train_direct_wsi_rna_survival as train_mod
from r2wsp.models.direct_survival import DirectWSIRNASurvival


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_effective_variant_and_gate(args_ns: argparse.Namespace) -> tuple[str, bool | None]:
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
    return model_variant, gate_enabled


def _flatten_norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).norm(dim=1)


def _flatten_mean(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1)


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-8)


def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    flat = weights.reshape(weights.shape[0], -1).clamp_min(1e-8)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ent = -(flat * flat.log()).sum(dim=1)
    max_ent = torch.log(torch.tensor(float(flat.shape[1]), device=flat.device)).clamp_min(1e-8)
    return ent / max_ent


def _mean_std_max(values: torch.Tensor, prefix: str, row: dict[str, object], idx: int) -> None:
    if values.ndim == 1:
        seq = values[idx]
        row[f"{prefix}_mean"] = float(seq)
        row[f"{prefix}_std"] = 0.0
        row[f"{prefix}_max"] = float(seq)
        return
    seq = values[idx]
    row[f"{prefix}_mean"] = float(seq.mean())
    row[f"{prefix}_std"] = float(seq.std(unbiased=False))
    row[f"{prefix}_max"] = float(seq.max())


def _append_curve_stats(row: dict[str, object], prefix: str, aux_cpu: dict[str, torch.Tensor], idx: int) -> None:
    points_key = f"{prefix}_points"
    if points_key not in aux_cpu:
        return
    points = aux_cpu[points_key][idx]
    if points.ndim != 2 or points.shape[0] < 2:
        return
    deltas = points[1:] - points[:-1]
    step_norm = deltas.norm(dim=1)
    centered = points - points.mean(dim=0, keepdim=True)
    bbox = points.max(dim=0).values - points.min(dim=0).values
    row[f"{prefix}_num_points"] = int(points.shape[0])
    row[f"{prefix}_path_length"] = float(step_norm.sum())
    row[f"{prefix}_endpoint_disp"] = float((points[-1] - points[0]).norm())
    row[f"{prefix}_step_mean"] = float(step_norm.mean())
    row[f"{prefix}_step_std"] = float(step_norm.std(unbiased=False))
    row[f"{prefix}_step_max"] = float(step_norm.max())
    row[f"{prefix}_point_var"] = float(points.var(dim=0, unbiased=False).mean())
    row[f"{prefix}_radius_mean"] = float(centered.norm(dim=1).mean())
    row[f"{prefix}_bbox_mean"] = float(bbox.mean())

    for suffix in ("curvature_per_point", "torsion_per_point"):
        key = f"{prefix}_{suffix}"
        if key in aux_cpu:
            stat_prefix = f"{prefix}_{suffix.replace('_per_point', '')}"
            _mean_std_max(aux_cpu[key], stat_prefix, row, idx)
    enhanced_key = f"{prefix}_geometry_enhanced"
    if enhanced_key in aux_cpu:
        row[f"{prefix}_enhanced_norm"] = float(_flatten_norm(aux_cpu[enhanced_key][idx : idx + 1])[0])


def _geometry_prefixes_from_aux(aux_cpu: dict[str, torch.Tensor]) -> list[str]:
    prefixes: set[str] = set()
    for key in aux_cpu.keys():
        if not (key.startswith("wsi_") or key.startswith("rna_")):
            continue
        if key.endswith("_points") and "points_by_slide" not in key:
            prefixes.add(key[: -len("_points")])
    return sorted(prefixes)


def _geometry_tensor_keys(aux_cpu: dict[str, torch.Tensor]) -> list[str]:
    keep: list[str] = []
    geometry_prefixes = _geometry_prefixes_from_aux(aux_cpu)
    fixed = {
        "cross_modal_interaction",
        "wsi_case",
        "rna_case",
        "fused",
        "wsi_geo_parallel",
        "rna_geo_fused",
    }
    for key in fixed:
        if key in aux_cpu:
            keep.append(key)
    for prefix in geometry_prefixes:
        for key in aux_cpu.keys():
            if key == prefix or key.startswith(f"{prefix}_"):
                keep.append(key)
    return sorted(set(keep))


def _aggregate_numeric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    out: dict[str, float] = {}
    for key, values in numeric.items():
        arr = np.asarray(values, dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
    return out


def _infer_fold(split_dir: str) -> int | None:
    match = re.search(r"k=(\d+)", str(split_dir))
    return int(match.group(1)) if match else None


def _infer_seed(run_dir: Path, args_ns: argparse.Namespace) -> int | None:
    if hasattr(args_ns, "seed"):
        try:
            return int(getattr(args_ns, "seed"))
        except (TypeError, ValueError):
            pass
    match = re.search(r"seed(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def _build_artifacts_from_run(
    args_ns: argparse.Namespace,
    *,
    export_batch_size: int | None,
) -> tuple[DirectWSIRNASurvival, dict[str, torch.utils.data.DataLoader], dict[str, dict[str, float]], torch.device]:
    model_variant, gate_enabled = _resolve_effective_variant_and_gate(args_ns)
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
    split_dir = Path(str(getattr(args_ns, "split_dir"))).resolve()
    cohort_hint = train_mod.infer_cohort_from_split_dir(split_dir)
    rows = train_mod.apply_cohort_hint(
        all_rows if bool(getattr(args_ns, "use_multi_slide", False)) else train_mod.dedupe_rows_by_case(all_rows, source_name=str(getattr(args_ns, "wsi_feature_source"))),
        cohort_hint=cohort_hint,
        reference_rows=all_rows,
    )
    rows = [r for r in rows if str(r.wsi_feature_source) == str(getattr(args_ns, "wsi_feature_source"))]
    case_table, train_case_ids, test_case_ids = train_mod.load_official_split(split_dir, str(getattr(args_ns, "target_col", "dss_survival_days")))
    train_case_ids, val_case_ids = train_mod.split_train_val_case_ids(
        sorted(train_case_ids),
        int(getattr(args_ns, "seed", 0)),
        float(getattr(args_ns, "val_frac", 0.2)),
        case_table,
        mode=str(getattr(args_ns, "val_split_mode", "survival_stratified")),
        n_time_bins=int(getattr(args_ns, "val_time_bins", 4)),
    )
    split_rows = {
        "train": [r for r in rows if str(r.case_id) in train_case_ids],
        "val": [r for r in rows if str(r.case_id) in val_case_ids],
        "test": [r for r in rows if str(r.case_id) in test_case_ids],
    }
    batch_size = int(export_batch_size if export_batch_size is not None else getattr(args_ns, "batch_size", 8))
    loaders = {
        split: train_mod.build_loader(
            split_rows[split],
            seed=int(getattr(args_ns, "seed", 0)),
            batch_size=batch_size,
            max_tiles=int(getattr(args_ns, "max_tiles", 256)),
            rna_mode=dataset_rna_mode,
            gene_sets_csv=gene_sets_csv,
            gene_id_to_symbol=gene_id_to_symbol,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            use_multi_slide=bool(getattr(args_ns, "use_multi_slide", False)),
            multi_slide_tile_budget_mode=str(getattr(args_ns, "multi_slide_tile_budget_mode", "per_slide")),
        )
        for split in ("train", "val", "test")
    }

    sample_loader = next(loader for loader in loaders.values() if len(loader.dataset) > 0)
    sample_batch = next(iter(sample_loader))
    tile_dim = int(sample_batch.tile_tokens.shape[-1])
    rna_dim = int(sample_batch.rna_vec.shape[-1]) if sample_batch.rna_vec is not None else 1
    rna_omic_sizes = [int(x.shape[-1]) for x in sample_batch.rna_omics] if sample_batch.rna_omics is not None else None
    device_arg = str(getattr(args_ns, "device", "auto"))
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    model = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=int(getattr(args_ns, "hidden_dim", 256)),
        dropout=float(getattr(args_ns, "dropout", 0.15)),
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
    ).to(device)
    return model, loaders, case_table, device


def export_run(
    run_dir: Path,
    *,
    ckpt_path: Path | None,
    splits: list[str],
    export_batch_size: int | None,
    device_override: str | None,
) -> None:
    ckpt_path = (ckpt_path.resolve() if ckpt_path is not None else (run_dir / "best.pt")).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_ns = argparse.Namespace(**ckpt["args"])
    if device_override is not None:
        setattr(args_ns, "device", device_override)
    model, loaders, case_table, device = _build_artifacts_from_run(args_ns, export_batch_size=export_batch_size)
    model.load_state_dict(ckpt["model"])
    model.eval()

    summary_path = run_dir / "summary.json"
    run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    fold = _infer_fold(str(getattr(args_ns, "split_dir", "")))
    seed = _infer_seed(run_dir, args_ns)

    export_root = run_dir / "case_diagnostics"
    export_root.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, object] = {}

    for split in splits:
        loader = loaders[split]
        rows: list[dict[str, object]] = []
        raw_tensors: dict[str, list[torch.Tensor]] = {}
        case_ids_all: list[str] = []
        with torch.no_grad():
            for batch in loader:
                times, events = train_mod.gather_labels(batch.case_id, case_table, device)
                risk, aux = model(
                    tile_tokens=batch.tile_tokens.to(device),
                    tile_xy=batch.tile_xy.to(device),
                    tile_attn_mask=batch.tile_attn_mask.to(device),
                    slide_ids=batch.slide_ids.to(device),
                    rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                    rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
                )
                aux_cpu = {k: v.detach().cpu() for k, v in aux.items() if isinstance(v, torch.Tensor)}
                risk_cpu = risk.detach().cpu().reshape(-1)
                times_cpu = times.detach().cpu().reshape(-1)
                events_cpu = events.detach().cpu().reshape(-1)
                wsi_case = aux_cpu["wsi_case"]
                rna_case = aux_cpu["rna_case"]
                fused = aux_cpu["fused"]
                gate = aux_cpu["gate"]
                wsi_norm = _flatten_norm(wsi_case)
                rna_norm = _flatten_norm(rna_case)
                fused_norm = _flatten_norm(fused)
                total_norm = wsi_norm + rna_norm

                for idx, case_id in enumerate(batch.case_id):
                    row: dict[str, object] = {
                        "run_name": run_dir.name,
                        "split": split,
                        "fold": fold,
                        "seed": seed,
                        "case_id": str(case_id),
                        "risk": float(risk_cpu[idx]),
                        "event_time": float(times_cpu[idx]),
                        "event": float(events_cpu[idx]),
                        "censorship": float(1.0 - events_cpu[idx]),
                        "wsi_case_norm": float(wsi_norm[idx]),
                        "rna_case_norm": float(rna_norm[idx]),
                        "fused_norm": float(fused_norm[idx]),
                        "wsi_norm_share": float(_safe_ratio(wsi_norm[idx], total_norm[idx])),
                        "rna_norm_share": float(_safe_ratio(rna_norm[idx], total_norm[idx])),
                        "gate_mean": float(_flatten_mean(gate[idx : idx + 1])[0]),
                        "gate_norm": float(_flatten_norm(gate[idx : idx + 1])[0]),
                    }
                    if "cross_modal_gate" in aux_cpu:
                        row["cross_modal_gate_mean"] = float(_flatten_mean(aux_cpu["cross_modal_gate"][idx : idx + 1])[0])
                    if "cross_modal_interaction" in aux_cpu:
                        row["cross_modal_interaction_norm"] = float(_flatten_norm(aux_cpu["cross_modal_interaction"][idx : idx + 1])[0])
                    if "cross_modal_wsi_case_updated" in aux_cpu:
                        delta = aux_cpu["cross_modal_wsi_case_updated"][idx : idx + 1] - wsi_case[idx : idx + 1]
                        row["cross_modal_wsi_delta_norm"] = float(_flatten_norm(delta)[0])
                    if "cross_modal_rna_case_updated" in aux_cpu:
                        delta = aux_cpu["cross_modal_rna_case_updated"][idx : idx + 1] - rna_case[idx : idx + 1]
                        row["cross_modal_rna_delta_norm"] = float(_flatten_norm(delta)[0])
                    if "slide_attn_weights" in aux_cpu:
                        slide_weights = aux_cpu["slide_attn_weights"][idx : idx + 1]
                        row["slide_attn_max"] = float(slide_weights.max())
                        row["slide_attn_entropy_norm"] = float(_normalized_entropy(slide_weights)[0])
                    if "tile_pool_weights" in aux_cpu:
                        tile_weights = aux_cpu["tile_pool_weights"][idx : idx + 1]
                        row["tile_pool_max"] = float(tile_weights.max())
                        row["tile_pool_entropy_norm"] = float(_normalized_entropy(tile_weights)[0])
                    if "rna_omics_attn_weights" in aux_cpu:
                        omics_weights = aux_cpu["rna_omics_attn_weights"][idx : idx + 1]
                        row["rna_omics_attn_max"] = float(omics_weights.max())
                        row["rna_omics_attn_entropy_norm"] = float(_normalized_entropy(omics_weights)[0])
                    for prefix in _geometry_prefixes_from_aux(aux_cpu):
                        _append_curve_stats(row, prefix, aux_cpu, idx)
                    rows.append(row)
                    case_ids_all.append(str(case_id))

                tensor_keys = _geometry_tensor_keys(aux_cpu)
                for key in tensor_keys:
                    if key in aux_cpu:
                        raw_tensors.setdefault(key, []).append(aux_cpu[key])
                raw_tensors.setdefault("risk", []).append(risk_cpu)
                raw_tensors.setdefault("event_time", []).append(times_cpu)
                raw_tensors.setdefault("event", []).append(events_cpu)

        fieldnames = list(rows[0].keys()) if rows else ["case_id"]
        _write_csv(export_root / f"{split}_cases.csv", rows, fieldnames)
        stacked = {key: torch.cat(value, dim=0) for key, value in raw_tensors.items()}
        stacked["case_id"] = case_ids_all
        torch.save(stacked, export_root / f"{split}_tensors.pt")
        split_summaries[split] = {
            "n_cases": len(rows),
            **_aggregate_numeric_summary(rows),
        }

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "run_summary": run_summary,
        "fold": fold,
        "seed": seed,
        "splits": splits,
        "export_batch_size": int(export_batch_size) if export_batch_size is not None else int(getattr(args_ns, "batch_size", 8)),
        "device": str(device),
        "summary": split_summaries,
    }
    _save_json(export_root / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-case predictions and geometry diagnostics from a trained direct-input survival run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt_path", default=None, help="Optional explicit checkpoint path. Default: <run_dir>/best.pt")
    parser.add_argument("--splits", default="test", help="Comma-separated splits: train,val,test")
    parser.add_argument("--export_batch_size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Optional override for export device.")
    args = parser.parse_args()

    splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    valid = {"train", "val", "test"}
    if not splits or any(x not in valid for x in splits):
        raise ValueError("splits must be a comma-separated subset of train,val,test")
    export_run(
        Path(args.run_dir).resolve(),
        ckpt_path=None if args.ckpt_path is None else Path(args.ckpt_path),
        splits=splits,
        export_batch_size=args.export_batch_size,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
