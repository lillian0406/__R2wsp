from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_mmp_latents import (  # noqa: E402
    PROTO_MODELS,
    _add_mmp_import_path,
    _build_dataloaders,
    _build_metadata,
    _export_latents,
    _find_single_run_dir,
    _load_checkpoint,
    _load_json,
    _prepare_args,
    _shape_or_none,
    _to_namespace,
)


def parse_folds(value: str) -> list[int]:
    folds = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not folds:
        raise ValueError("at least one fold is required")
    return folds


def torch_load_compat(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_export_arrays(path: Path) -> dict[str, dict[str, np.ndarray]]:
    obj = torch_load_compat(path)
    if not isinstance(obj, dict):
        raise TypeError(f"invalid export payload: {path}")
    export: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_obj in obj.items():
        if not isinstance(split_obj, dict):
            raise TypeError(f"invalid split payload {split_name} in {path}")
        export[split_name] = {key: _to_numpy(value) for key, value in split_obj.items()}
    return export


def _concat_object(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=object)
    return np.concatenate(values, axis=0).astype(object, copy=False)


def _concat_numeric(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(values, axis=0)


def _build_oof_bundle(export_root: Path, folds: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    combined: dict[str, list[np.ndarray]] = {}
    sample_ids_by_fold: dict[int, list[str]] = {}

    for fold in folds:
        export_path = export_root / f"k={fold}" / "mmp_latent_export.pt"
        export = _load_export_arrays(export_path)
        test_obj = export["test"]
        sample_ids = test_obj["sample_ids"].astype(object)
        sample_ids_by_fold[fold] = [str(x) for x in sample_ids.tolist()]
        for key, value in test_obj.items():
            combined.setdefault(key, []).append(value)
        combined.setdefault("fold_index", []).append(np.full(sample_ids.shape[0], fold, dtype=np.int64))

    oof_payload = {
        "sample_ids": _concat_object(combined.get("sample_ids", [])),
        "event_times": _concat_numeric(combined.get("event_times", [])),
        "censorships": _concat_numeric(combined.get("censorships", [])),
        "risk_scores": _concat_numeric(combined.get("risk_scores", [])),
        "logits": _concat_numeric(combined.get("logits", [])),
        "wsi_embedding": _concat_numeric(combined.get("wsi_embedding", [])),
        "rna_embedding": _concat_numeric(combined.get("rna_embedding", [])),
        "pathway_embedding": _concat_numeric(combined.get("pathway_embedding", [])),
        "fusion_embedding": _concat_numeric(combined.get("fusion_embedding", [])),
        "risk_head_input": _concat_numeric(combined.get("risk_head_input", [])),
        "fold_index": _concat_numeric(combined.get("fold_index", [])),
    }

    seen: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []
    for fold, sample_ids in sample_ids_by_fold.items():
        for sample_id in sample_ids:
            if sample_id in seen:
                duplicates.append({"sample_id": sample_id, "first_fold": seen[sample_id], "duplicate_fold": fold})
            else:
                seen[sample_id] = fold

    manifest = {
        "folds": folds,
        "num_oof_samples": int(len(oof_payload["sample_ids"])),
        "duplicate_oof_sample_ids": duplicates[:50],
        "num_duplicate_oof_sample_ids": len(duplicates),
        "oof_feature_shapes": {key: _shape_or_none(value) for key, value in oof_payload.items()},
        "test_sample_counts_by_fold": {f"k={fold}": len(sample_ids_by_fold[fold]) for fold in folds},
    }
    return oof_payload, manifest


def _export_one_fold(
    *,
    run_dir: Path,
    export_dir: Path,
    num_workers: int,
    batch_size: int | None,
    include_token_level: bool,
    include_attention: bool,
) -> dict[str, Any]:
    checkpoint_path = (run_dir / "s_checkpoint.pth").resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    run_config = _to_namespace(_load_json(run_dir / "config.json"))
    run_config = _prepare_args(run_config)
    run_config.num_workers = int(num_workers)
    if batch_size is not None:
        run_config.batch_size = int(batch_size)

    if str(run_config.model_histo_type) in PROTO_MODELS:
        raise NotImplementedError("export_mmp_latents_5fold.py currently supports non-prototype histology models only.")
    if str(run_config.model_mm_type).lower() not in {"survpath", "coattn", "histo", "gene"}:
        raise NotImplementedError(f"unsupported model_mm_type for latent export: {run_config.model_mm_type}")

    from utils.utils import seed_torch
    from mil_models import create_multimodal_survival_model

    seed_torch(int(getattr(run_config, "seed", 1)))
    loaders, _ = _build_dataloaders(run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config.feat_dim = int(run_config.in_dim)
    model = create_multimodal_survival_model(run_config, omic_sizes=loaders["train"].dataset.omic_sizes).to(device)
    _load_checkpoint(model, checkpoint_path)

    export = _export_latents(
        model=model,
        loaders=loaders,
        device=device,
        loss_fn_name=str(run_config.loss_fn),
        include_token_level=bool(include_token_level),
        include_attention=bool(include_attention),
    )

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "mmp_latent_export.pt"
    torch.save(export, export_path)

    metadata = _build_metadata(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        export_path=export_path,
        args=run_config,
        export=export,
    )
    metadata_path = export_dir / "mmp_latent_export.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description="Export a fair 5-fold latent benchmark from frozen MMP runs.")
    p.add_argument("--mmp-root", default="/root/autodl-tmp/_refs/MMP-main")
    p.add_argument("--results-root", required=True, help="Directory that contains k=0 ... k=4 MMP result folders.")
    p.add_argument("--export-root", required=True, help="Output root for k=<fold> latent exports and manifest files.")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--include-token-level", action="store_true", default=False)
    p.add_argument("--include-attention", action="store_true", default=False)
    args = p.parse_args()

    folds = parse_folds(args.folds)
    mmp_root = Path(args.mmp_root).resolve()
    results_root = Path(args.results_root).resolve()
    export_root = Path(args.export_root).resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    mmp_src = _add_mmp_import_path(mmp_root)
    os.chdir(mmp_src)

    per_fold_metadata: dict[str, Any] = {}
    for fold in folds:
        fold_results_dir = results_root / f"k={fold}"
        run_dir = _find_single_run_dir(None, fold_results_dir)
        fold_export_dir = export_root / f"k={fold}"
        metadata = _export_one_fold(
            run_dir=run_dir,
            export_dir=fold_export_dir,
            num_workers=int(args.num_workers),
            batch_size=args.batch_size,
            include_token_level=bool(args.include_token_level),
            include_attention=bool(args.include_attention),
        )
        per_fold_metadata[f"k={fold}"] = metadata
        print(f"[done] k={fold} -> {fold_export_dir}")

    oof_payload, oof_manifest = _build_oof_bundle(export_root, folds)
    oof_path = export_root / "mmp_latent_oof_test.pt"
    torch.save(oof_payload, oof_path)

    manifest = {
        "results_root": str(results_root),
        "export_root": str(export_root),
        "folds": folds,
        "per_fold_metadata": per_fold_metadata,
        "oof_test_bundle": {
            "path": str(oof_path),
            **oof_manifest,
        },
    }
    manifest_path = export_root / "mmp_latent_5fold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"manifest_path={manifest_path}")
    print(f"oof_test_path={oof_path}")
    print(f"oof_samples={oof_manifest['num_oof_samples']}")
    print(f"oof_duplicates={oof_manifest['num_duplicate_oof_sample_ids']}")


if __name__ == "__main__":
    main()
