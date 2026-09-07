from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROTO_MODELS = {"PANTHER", "OT", "H2T", "ProtoCount"}


class _ArgsNamespace(SimpleNamespace):
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


def _add_mmp_import_path(mmp_root: Path) -> Path:
    mmp_src = (mmp_root / "src").resolve()
    if not mmp_src.exists():
        raise FileNotFoundError(mmp_src)
    if str(mmp_src) not in sys.path:
        sys.path.insert(0, str(mmp_src))
    return mmp_src


def _find_single_run_dir(run_dir: Path | None, fold_results_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = run_dir.resolve()
        if not (resolved / "config.json").exists():
            raise FileNotFoundError(resolved / "config.json")
        return resolved

    if fold_results_dir is None:
        raise ValueError("either --run-dir or --fold-results-dir is required")

    candidates = sorted(
        p.parent for p in fold_results_dir.resolve().glob("**/config.json") if (p.parent / "s_checkpoint.pth").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no run directory with config.json + s_checkpoint.pth found under {fold_results_dir}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_namespace(config: dict[str, Any]) -> SimpleNamespace:
    args = _ArgsNamespace(**config)
    if isinstance(getattr(args, "data_source", None), str):
        args.data_source = [args.data_source]
    # Keep split_names as the original comma-separated string because MMP's
    # utils.read_splits() expects args.split_names.split(',').
    split_names = getattr(args, "split_names", None)
    if isinstance(split_names, list):
        args.split_names = ",".join(str(x).strip() for x in split_names if str(x).strip())
    elif not split_names:
        args.split_names = "train,test"
    return args


def _maybe_expand_omics_dir(omics_dir: Path, *, type_of_path: str, cancer_type: str) -> Path:
    candidate = omics_dir / type_of_path / cancer_type
    if candidate.exists():
        return candidate.resolve()
    return omics_dir.resolve()


def _prepare_args(args: SimpleNamespace) -> SimpleNamespace:
    if getattr(args, "train_bag_size", -1) == -1:
        args.train_bag_size = args.bag_size
    if getattr(args, "val_bag_size", -1) == -1:
        args.val_bag_size = args.bag_size
    if getattr(args, "loss_fn", "nll") != "nll":
        args.n_label_bins = 0
    args.data_source = [str(Path(src).resolve()) for src in args.data_source]
    split_dir = Path(args.split_dir).resolve()
    args.split_dir = str(split_dir)
    cancer_type = split_dir.name.split("_")[1]
    args.omics_dir = str(
        _maybe_expand_omics_dir(
            Path(args.omics_dir),
            type_of_path=str(getattr(args, "type_of_path", "hallmarks")),
            cancer_type=cancer_type,
        )
    )
    return args


def _build_dataset_kwargs(args: SimpleNamespace) -> tuple[dict[str, Any], dict[str, Any]]:
    censorship_col = str(args.target_col).split("_")[0] + "_censorship"
    common = dict(
        data_source=args.data_source,
        survival_time_col=args.target_col,
        censorship_col=censorship_col,
        n_label_bins=args.n_label_bins,
        label_bins=None,
        omics_dir=args.omics_dir,
        omics_modality=args.omics_modality,
    )
    train_kwargs = dict(common)
    # Export is read-only inference, so keep a stable order for train split as well.
    train_kwargs.update(bag_size=args.train_bag_size, shuffle=False)
    val_kwargs = dict(common)
    val_kwargs.update(bag_size=args.val_bag_size, shuffle=False)
    return train_kwargs, val_kwargs


def _build_dataloaders(args: SimpleNamespace) -> tuple[dict[str, DataLoader], Any]:
    from utils.utils import read_splits
    from wsi_datasets import WSIOmicsSurvivalDataset

    csv_splits = read_splits(args)
    train_kwargs, val_kwargs = _build_dataset_kwargs(args)

    dataset_splits: dict[str, DataLoader] = {}
    label_bins = None
    scaler = None
    for split_name in csv_splits.keys():
        df = csv_splits[split_name]
        dataset_kwargs = train_kwargs.copy() if split_name == "train" else val_kwargs.copy()
        dataset_kwargs["label_bins"] = label_bins
        dataset = WSIOmicsSurvivalDataset(df_histo=df["histo"], df_gene=df["gene"], **dataset_kwargs)
        batch_size = int(args.batch_size)
        if str(args.model_histo_type) not in PROTO_MODELS:
            batch_size = batch_size if int(dataset_kwargs.get("bag_size", -1)) > 0 else 1
        if split_name == "train":
            scaler = dataset.get_scaler()
        assert scaler is not None, "omics scaler from train split is required"
        dataset.apply_scaler(scaler)
        dataset_splits[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(dataset_kwargs["shuffle"]),
            num_workers=int(args.num_workers),
        )
        if getattr(args, "loss_fn", "nll") == "nll" and split_name == "train":
            label_bins = dataset.get_label_bins()
    return dataset_splits, scaler


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def _extract_coattn_latents(model: torch.nn.Module, x_path: torch.Tensor, x_omics: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    from mil_models.model_multimodal import agg_histo

    device = x_path.device

    h_omic = []
    for idx, sig_feat in enumerate(x_omics):
        omic_feat = model.sig_networks[idx](sig_feat.float())
        h_omic.append(omic_feat)
    h_omic = torch.stack(h_omic, dim=1)

    if getattr(model, "gene_embedding", None) is not None:
        gene_embedding = model.gene_embedding.to(device)
        h_omic = torch.cat([torch.cat([h_omic[idx : idx + 1], gene_embedding], dim=-1) for idx in range(len(h_omic))], dim=0)

    h_path = model.path_proj_net(x_path)
    if getattr(model, "histo_embedding", None) is not None:
        histo_embedding = model.histo_embedding.to(device)
        h_path = torch.cat([torch.cat([h_path[idx : idx + 1], histo_embedding], dim=-1) for idx in range(len(h_path))], dim=0)

    tokens = torch.cat([h_omic, h_path], dim=1)
    tokens = model.identity(tokens)
    mm_tokens = model.coattn(tokens)

    pathway_tokens = mm_tokens[:, : model.num_pathways, :]
    pathway_embedding = torch.mean(pathway_tokens, dim=1)

    wsi_tokens = mm_tokens[:, model.num_pathways :, :]
    if str(getattr(model, "histo_model", "")).lower() == "mil":
        wsi_embedding = torch.mean(wsi_tokens, dim=1)
    else:
        wsi_embedding = agg_histo(wsi_tokens, model.histo_agg)

    modality = str(getattr(model, "modality", "both")).lower()
    if modality == "histo":
        fusion_embedding = wsi_embedding
    elif modality == "gene":
        fusion_embedding = pathway_embedding
    else:
        fusion_embedding = torch.cat([pathway_embedding, wsi_embedding], dim=1)

    logits = model.classifier(fusion_embedding)
    return {
        "pathway_embedding": pathway_embedding,
        "wsi_embedding": wsi_embedding,
        "fusion_embedding": fusion_embedding,
        "risk_head_input": fusion_embedding,
        "logits_from_latent": logits,
        "pathway_tokens_pre_fusion": h_omic,
        "wsi_tokens_pre_fusion": h_path,
        "multimodal_tokens_post_fusion": mm_tokens,
        "pathway_tokens_post_fusion": pathway_tokens,
        "wsi_tokens_post_fusion": wsi_tokens,
    }


def _collect_attentions(model: torch.nn.Module, x_path: torch.Tensor, x_omics: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    device = x_path.device
    h_omic = []
    for idx, sig_feat in enumerate(x_omics):
        omic_feat = model.sig_networks[idx](sig_feat.float())
        h_omic.append(omic_feat)
    h_omic = torch.stack(h_omic, dim=1)
    if getattr(model, "gene_embedding", None) is not None:
        gene_embedding = model.gene_embedding.to(device)
        h_omic = torch.cat([torch.cat([h_omic[idx : idx + 1], gene_embedding], dim=-1) for idx in range(len(h_omic))], dim=0)

    h_path = model.path_proj_net(x_path)
    if getattr(model, "histo_embedding", None) is not None:
        histo_embedding = model.histo_embedding.to(device)
        h_path = torch.cat([torch.cat([h_path[idx : idx + 1], histo_embedding], dim=-1) for idx in range(len(h_path))], dim=0)

    tokens = torch.cat([h_omic, h_path], dim=1)
    tokens = model.identity(tokens)
    _, omic_attn, cross_attn, path_attn = model.coattn[0](x=tokens, mask=None, return_attention=True)
    return {
        "omic_attn": omic_attn,
        "cross_attn": cross_attn,
        "path_attn": path_attn,
    }


def _empty_split_export() -> dict[str, list[Any]]:
    return {
        "sample_ids": [],
        "event_times": [],
        "censorships": [],
        "risk_scores": [],
        "logits": [],
        "wsi_embedding": [],
        "rna_embedding": [],
        "pathway_embedding": [],
        "fusion_embedding": [],
        "risk_head_input": [],
    }


def _compute_risk_scores(logits: torch.Tensor, *, loss_fn_name: str) -> torch.Tensor:
    mode = str(loss_fn_name).lower()
    if mode == "nll":
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return -torch.sum(survival, dim=1, keepdim=True)
    if mode == "cox":
        return torch.exp(logits)
    if mode == "rank":
        return logits
    raise NotImplementedError(f"unsupported loss_fn for risk export: {loss_fn_name}")


def _append_batch(
    store: dict[str, list[Any]],
    *,
    batch: dict[str, Any],
    logits: torch.Tensor,
    risk_scores: torch.Tensor,
    latents: dict[str, torch.Tensor],
) -> None:
    sample_ids = batch["sample_ids"]
    store["sample_ids"].extend([str(x) for x in sample_ids])
    store["event_times"].append(batch["survival_time"].detach().cpu())
    store["censorships"].append(batch["censorship"].detach().cpu())
    store["risk_scores"].append(risk_scores.detach().cpu())
    store["logits"].append(logits.detach().cpu())
    store["wsi_embedding"].append(latents["wsi_embedding"].detach().cpu())
    store["rna_embedding"].append(latents["pathway_embedding"].detach().cpu())
    store["pathway_embedding"].append(latents["pathway_embedding"].detach().cpu())
    store["fusion_embedding"].append(latents["fusion_embedding"].detach().cpu())
    store["risk_head_input"].append(latents["risk_head_input"].detach().cpu())


def _finalize_split_export(store: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "sample_ids": np.asarray(store["sample_ids"], dtype=object),
        "event_times": torch.cat(store["event_times"], dim=0).squeeze(1).numpy() if store["event_times"] else np.empty((0,)),
        "censorships": torch.cat(store["censorships"], dim=0).squeeze(1).numpy() if store["censorships"] else np.empty((0,)),
        "risk_scores": torch.cat(store["risk_scores"], dim=0).squeeze(1).numpy() if store["risk_scores"] else np.empty((0,)),
        "logits": torch.cat(store["logits"], dim=0).numpy() if store["logits"] else np.empty((0,)),
        "wsi_embedding": torch.cat(store["wsi_embedding"], dim=0).numpy() if store["wsi_embedding"] else np.empty((0,)),
        "rna_embedding": torch.cat(store["rna_embedding"], dim=0).numpy() if store["rna_embedding"] else np.empty((0,)),
        "pathway_embedding": torch.cat(store["pathway_embedding"], dim=0).numpy() if store["pathway_embedding"] else np.empty((0,)),
        "fusion_embedding": torch.cat(store["fusion_embedding"], dim=0).numpy() if store["fusion_embedding"] else np.empty((0,)),
        "risk_head_input": torch.cat(store["risk_head_input"], dim=0).numpy() if store["risk_head_input"] else np.empty((0,)),
    }


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _export_latents(
    *,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    loss_fn_name: str,
    include_token_level: bool,
    include_attention: bool,
) -> dict[str, Any]:
    from utils.utils import safe_list_to

    model.eval()
    export: dict[str, Any] = {}
    with torch.no_grad():
        for split_name, loader in loaders.items():
            split_store = _empty_split_export()
            token_level: dict[str, list[np.ndarray]] = {
                "pathway_tokens_pre_fusion": [],
                "wsi_tokens_pre_fusion": [],
                "multimodal_tokens_post_fusion": [],
                "pathway_tokens_post_fusion": [],
                "wsi_tokens_post_fusion": [],
            }
            attn_store: dict[str, list[np.ndarray]] = {
                "omic_attn": [],
                "cross_attn": [],
                "path_attn": [],
            }
            for batch_idx, batch in enumerate(loader):
                data = batch["img"].to(device)
                omics = safe_list_to(batch["omics"], device)
                latents = _extract_coattn_latents(model, data, omics)
                logits = latents["logits_from_latent"]
                risk_scores = _compute_risk_scores(logits, loss_fn_name=loss_fn_name)
                batch["sample_ids"] = loader.dataset.idx2sample_df.iloc[
                    batch_idx * len(data) : batch_idx * len(data) + len(data)
                ]["sample_id"].astype(str).tolist()
                _append_batch(
                    split_store,
                    batch=batch,
                    logits=logits,
                    risk_scores=risk_scores,
                    latents=latents,
                )

                if include_token_level:
                    for key in token_level:
                        token_level[key].append(_tensor_to_numpy(latents[key]))

                if include_attention:
                    attn = _collect_attentions(model, data, omics)
                    for key in attn_store:
                        attn_store[key].append(_tensor_to_numpy(attn[key]))

            split_export = _finalize_split_export(split_store)
            if include_token_level:
                split_export.update({k: np.concatenate(v, axis=0) if v else np.empty((0,)) for k, v in token_level.items()})
            if include_attention:
                split_export.update({k: np.concatenate(v, axis=0) if v else np.empty((0,)) for k, v in attn_store.items()})
            export[split_name] = split_export
    return export


def _shape_or_none(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def _build_metadata(*, run_dir: Path, checkpoint_path: Path, export_path: Path, args: SimpleNamespace, export: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "source_run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "split_dir": str(args.split_dir),
        "data_source": list(args.data_source),
        "omics_dir": str(args.omics_dir),
        "model_histo_type": str(args.model_histo_type),
        "model_mm_type": str(args.model_mm_type),
        "target_col": str(args.target_col),
        "export_path": str(export_path),
        "splits": {},
    }
    for split_name, split_data in export.items():
        metadata["splits"][split_name] = {key: _shape_or_none(value) for key, value in split_data.items()}
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description="Export richer patient-level MMP latents without modifying the MMP source tree.")
    p.add_argument("--mmp-root", default="/root/autodl-tmp/_refs/MMP-main")
    p.add_argument("--fold-results-dir", default=None, help="Directory like .../results_5fold/k=0; latest nested run is auto-detected.")
    p.add_argument("--run-dir", default=None, help="Exact nested MMP run directory that contains config.json and s_checkpoint.pth.")
    p.add_argument("--export-dir", required=True, help="Output directory for the read-only latent package.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None, help="Override export batch size. Defaults to the run config value.")
    p.add_argument("--include-token-level", action="store_true", default=False)
    p.add_argument("--include-attention", action="store_true", default=False)
    args = p.parse_args()

    mmp_root = Path(args.mmp_root).resolve()
    run_dir = _find_single_run_dir(
        Path(args.run_dir).resolve() if args.run_dir else None,
        Path(args.fold_results_dir).resolve() if args.fold_results_dir else None,
    )
    checkpoint_path = (run_dir / "s_checkpoint.pth").resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    mmp_src = _add_mmp_import_path(mmp_root)
    # MMP uses relative signature paths like ./data_csvs/rna/metadata inside the dataset code.
    os.chdir(mmp_src)

    run_config = _to_namespace(_load_json(run_dir / "config.json"))
    run_config = _prepare_args(run_config)
    run_config.num_workers = int(args.num_workers)
    if args.batch_size is not None:
        run_config.batch_size = int(args.batch_size)

    if str(run_config.model_histo_type) in PROTO_MODELS:
        raise NotImplementedError("export_mmp_latents.py currently supports non-prototype histology models only.")
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
        include_token_level=bool(args.include_token_level),
        include_attention=bool(args.include_attention),
    )

    export_dir = Path(args.export_dir).resolve()
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

    print("run_dir       =", run_dir)
    print("checkpoint    =", checkpoint_path)
    print("export_path   =", export_path)
    print("metadata_path =", metadata_path)
    for split_name, split_data in export.items():
        print(f"[{split_name}] samples={len(split_data['sample_ids'])}")
        for key in ["risk_scores", "wsi_embedding", "rna_embedding", "fusion_embedding", "pathway_embedding"]:
            print(f"[{split_name}] {key} shape={_shape_or_none(split_data[key])}")


if __name__ == "__main__":
    main()
