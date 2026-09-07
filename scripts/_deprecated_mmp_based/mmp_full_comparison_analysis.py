from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


def torch_load_compat(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def concordance_index(risk: np.ndarray, times: np.ndarray, censorships: np.ndarray) -> float:
    events = 1.0 - censorships.astype(np.float32)
    concordant = 0.0
    comparable = 0.0
    n = int(risk.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and events[i] > 0:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] > 0:
                comparable += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return x / norms


def cosine_similarity_report(train_x: np.ndarray, test_x: np.ndarray) -> dict[str, object]:
    train_n = normalize_rows(train_x)
    test_n = normalize_rows(test_x)
    sims = test_n @ train_n.T
    max_sim = sims.max(axis=1)
    return {
        "mean_max_cosine_similarity": float(np.mean(max_sim)),
        "median_max_cosine_similarity": float(np.median(max_sim)),
        "max_cosine_similarity": float(np.max(max_sim)),
        "num_test_ge_0.9999": int(np.sum(max_sim >= 0.9999)),
        "num_test_ge_0.999": int(np.sum(max_sim >= 0.999)),
        "num_test_ge_0.99": int(np.sum(max_sim >= 0.99)),
        "top5_test_to_train_max_similarity": [float(x) for x in np.sort(max_sim)[-5:][::-1]],
    }


def attention_entropy(attn: np.ndarray, axis: int = -1) -> np.ndarray:
    x = attn.astype(np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    p = np.exp(x)
    p = p / np.clip(p.sum(axis=axis, keepdims=True), 1e-12, None)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def attention_stats(name: str, attn_logits: np.ndarray, axis: int = -1) -> dict[str, object]:
    ent = attention_entropy(attn_logits, axis=axis)
    ent_flat = ent.reshape(-1)
    eff = np.exp(ent_flat)
    return {
        f"{name}_entropy_mean": float(ent_flat.mean()),
        f"{name}_entropy_std": float(ent_flat.std()),
        f"{name}_effective_k_mean": float(eff.mean()),
        f"{name}_effective_k_std": float(eff.std()),
        f"{name}_logit_mean": float(attn_logits.mean()),
        f"{name}_logit_std": float(attn_logits.std()),
    }


def load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df.reindex(columns=fieldnames)
    df.to_csv(path, index=False)


@dataclass
class MMPFoldRun:
    fold: int
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    test_results_path: Path
    train_results_path: Path | None
    val_results_path: Path | None


def discover_mmp_fold_runs(results_root: Path) -> list[MMPFoldRun]:
    runs: list[MMPFoldRun] = []
    for fold_dir in sorted(results_root.glob("k=*")):
        fold_str = fold_dir.name.split("=")[-1]
        try:
            fold = int(fold_str)
        except ValueError:
            continue
        candidates = list(fold_dir.glob("LUAD_survival/**/s_checkpoint.pth"))
        if not candidates:
            continue
        ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
        run_dir = ckpt.parent
        config_path = run_dir / "config.json"
        test_path = run_dir / "test_results.pkl"
        if not test_path.exists():
            test_path = run_dir / "test_results.pkl"
        train_path = run_dir / "train_results.pkl"
        if not train_path.exists():
            train_path = None
        val_path = run_dir / "val_results.pkl"
        if not val_path.exists():
            val_path = None
        runs.append(
            MMPFoldRun(
                fold=fold,
                run_dir=run_dir,
                checkpoint_path=ckpt,
                config_path=config_path,
                test_results_path=test_path,
                train_results_path=train_path,
                val_results_path=val_path,
            )
        )
    if len(runs) == 5:
        return sorted(runs, key=lambda r: r.fold)
    return sorted(runs, key=lambda r: r.fold)


def resolve_device(device_arg: str) -> torch.device:
    normalized = str(device_arg).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_arg}")


def import_mmp_src(mmp_src: Path) -> None:
    sys.path.insert(0, str(mmp_src))
    sys.path.insert(0, str(mmp_src.parent))
    if (mmp_src / "data_csvs" / "rna" / "metadata").is_dir():
        os.chdir(mmp_src)


def load_mmp_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(str(config_path))


def resolve_omics_dir(config: dict[str, Any]) -> dict[str, Any]:
    omics_dir = config.get("omics_dir")
    if not omics_dir:
        return config
    omics_path = Path(str(omics_dir))
    if (omics_path / "rna_clean.csv").is_file():
        return config

    task = str(config.get("task", "")).strip()
    cohort = task.split("_", 1)[0] if task else ""
    type_of_path = str(config.get("type_of_path", "")).strip()
    candidates: list[Path] = []
    if type_of_path and cohort:
        candidates.append(omics_path / type_of_path / cohort)
    if cohort:
        candidates.append(omics_path / "hallmarks" / cohort)

    for candidate in candidates:
        if (candidate / "rna_clean.csv").is_file():
            patched = dict(config)
            patched["omics_dir"] = str(candidate)
            return patched

    return config


def normalize_mmp_config(config: dict[str, Any]) -> dict[str, Any]:
    patched = dict(config)
    if "feat_dim" not in patched and "in_dim" in patched:
        patched["feat_dim"] = patched["in_dim"]
    return patched


def build_mmp_dataloaders_from_config(config: dict[str, Any]) -> dict[str, DataLoader]:
    from utils.utils import read_splits
    from wsi_datasets import WSIOmicsSurvivalDataset

    class ArgsObj:
        def __init__(self, d: dict[str, Any]) -> None:
            self.__dict__.update(d)
        
        def __contains__(self, key: str) -> bool:
            return hasattr(self, key)

    args = ArgsObj(config)
    csv_splits = read_splits(args)
    dataset_splits: dict[str, DataLoader] = {}
    label_bins = None
    scaler = None
    for split_name, df_pair in csv_splits.items():
        shuffle = True if split_name == "train" else False
        kwargs = dict(
            data_source=args.data_source,
            survival_time_col=args.target_col,
            censorship_col=args.target_col.split("_")[0] + "_censorship",
            n_label_bins=getattr(args, "n_label_bins", 0),
            label_bins=label_bins,
            bag_size=args.train_bag_size if split_name == "train" else args.val_bag_size,
            shuffle=shuffle,
            omics_dir=args.omics_dir,
            omics_modality=args.omics_modality,
        )
        dataset = WSIOmicsSurvivalDataset(df_histo=df_pair["histo"], df_gene=df_pair["gene"], **kwargs)
        if split_name == "train":
            scaler = dataset.get_scaler()
        if scaler is None:
            raise RuntimeError("Missing scaler from train split.")
        dataset.apply_scaler(scaler)
        if (getattr(args, "loss_fn", "nll") == "nll") and (split_name == "train"):
            label_bins = dataset.get_label_bins()
        batch_size = int(getattr(args, "batch_size", 1))
        if str(getattr(args, "model_histo_type", "")).upper() not in {"PANTHER", "OT", "H2T", "PROTOCOUNT"}:
            if int(kwargs.get("bag_size", -1)) <= 0:
                batch_size = 1
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(getattr(args, "num_workers", 2)))
        dataset_splits[split_name] = loader
    return dataset_splits


def build_mmp_model_from_config(config: dict[str, Any], omic_sizes: list[int]) -> nn.Module:
    from mil_models.model_factory import create_multimodal_survival_model

    class ArgsObj:
        def __init__(self, d: dict[str, Any]) -> None:
            self.__dict__.update(d)

        def __contains__(self, key: str) -> bool:
            return hasattr(self, key)

    args = ArgsObj(config)
    return create_multimodal_survival_model(args, omic_sizes=omic_sizes)


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    groups: dict[str, int] = {
        "total": 0,
        "sig_networks": 0,
        "path_proj_net": 0,
        "coattn": 0,
        "classifier": 0,
        "histo_embedding": 0,
        "gene_embedding": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        n = int(p.numel())
        groups["total"] += n
        if name.startswith("sig_networks."):
            groups["sig_networks"] += n
        elif name.startswith("path_proj_net."):
            groups["path_proj_net"] += n
        elif name.startswith("coattn."):
            groups["coattn"] += n
        elif name.startswith("classifier."):
            groups["classifier"] += n
        elif name.startswith("histo_embedding"):
            groups["histo_embedding"] += n
        elif name.startswith("gene_embedding"):
            groups["gene_embedding"] += n
        else:
            groups["other"] += n
    return groups


def extract_mmp_embeddings_and_attn(model: nn.Module, batch: dict[str, Any], *, device: torch.device, return_attn: bool) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_path = batch["img"]
        if isinstance(x_path, list):
            x_path = x_path[0]
        x_path = x_path.to(device)
        omics = batch["omics"]
        if isinstance(omics, list):
            omics = [x.to(device) for x in omics]
        else:
            omics = [x.to(device) for x in list(omics)]
        if hasattr(model, "forward_no_loss"):
            out = model.forward_no_loss(x_path, omics, return_attn=return_attn)
        else:
            out = model(x_path, omics, return_attn=return_attn)[0]
        logits = out["logits"]
        emb = None
        def hook_fn(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _outputs: torch.Tensor) -> None:
            nonlocal emb
            emb = inputs[0].detach()
        handle = None
        if hasattr(model, "classifier"):
            handle = model.classifier.register_forward_hook(hook_fn)
            _ = model.forward_no_loss(x_path, omics, return_attn=False) if hasattr(model, "forward_no_loss") else model(x_path, omics)[0]
            handle.remove()
        return {
            "logits": logits.detach(),
            "embedding": emb,
            "omic_attn": out.get("omic_attn"),
            "cross_attn": out.get("cross_attn"),
            "path_attn": out.get("path_attn"),
        }


def risk_from_logits(logits: torch.Tensor, loss_fn: str) -> torch.Tensor:
    if loss_fn == "nll":
        hazards = torch.softmax(logits, dim=1)
        risk = torch.cumsum(hazards, dim=1).sum(dim=1, keepdim=True)
        return risk
    return logits.reshape(-1, 1)


def evaluate_mmp_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: str,
    return_attn: bool,
) -> dict[str, Any]:
    all_risk: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    all_cens: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    all_omic_attn: list[np.ndarray] = []
    all_cross_attn: list[np.ndarray] = []
    all_path_attn: list[np.ndarray] = []
    emb_list: list[np.ndarray] = []
    sample_cursor = 0
    model.eval()
    for batch in loader:
        out = extract_mmp_embeddings_and_attn(model, batch, device=device, return_attn=return_attn)
        logits = out["logits"]
        risk = risk_from_logits(logits, loss_fn=str(loss_fn)).detach().cpu().numpy().reshape(-1)
        batch_size = int(risk.shape[0])
        all_risk.append(risk)
        all_times.append(batch["survival_time"].detach().cpu().numpy().reshape(-1))
        all_cens.append(batch["censorship"].detach().cpu().numpy().reshape(-1))
        if "sample_id" in batch:
            batch_ids = np.asarray([str(x) for x in batch["sample_id"]], dtype=str)
        elif hasattr(loader.dataset, "get_sample_id"):
            batch_ids = np.asarray(
                [str(loader.dataset.get_sample_id(i)) for i in range(sample_cursor, sample_cursor + batch_size)],
                dtype=str,
            )
        else:
            batch_ids = np.asarray([str(sample_cursor + i) for i in range(batch_size)], dtype=str)
        sample_cursor += batch_size
        all_ids.append(batch_ids)
        if out["embedding"] is not None:
            emb_list.append(out["embedding"].detach().cpu().numpy())
        if return_attn:
            if out["omic_attn"] is not None:
                all_omic_attn.append(np.asarray(out["omic_attn"]))
            if out["cross_attn"] is not None:
                all_cross_attn.append(np.asarray(out["cross_attn"]))
            if out["path_attn"] is not None:
                all_path_attn.append(np.asarray(out["path_attn"]))
    risk_all = np.concatenate(all_risk).astype(np.float32)
    times_all = np.concatenate(all_times).astype(np.float32)
    cens_all = np.concatenate(all_cens).astype(np.float32)
    ids_all = np.concatenate(all_ids).astype(str)
    payload: dict[str, Any] = {
        "risk": risk_all,
        "event_times": times_all,
        "censorships": cens_all,
        "sample_ids": ids_all,
        "c_index": concordance_index(risk_all, times_all, cens_all),
    }
    if emb_list:
        payload["embedding"] = np.concatenate(emb_list).astype(np.float32)
    if all_omic_attn:
        payload["omic_attn"] = np.concatenate(all_omic_attn).astype(np.float32)
    if all_cross_attn:
        payload["cross_attn"] = np.concatenate(all_cross_attn).astype(np.float32)
    if all_path_attn:
        payload["path_attn"] = np.concatenate(all_path_attn).astype(np.float32)
    return payload


def patch_mmp_disable_cross_attn(model: nn.Module) -> None:
    if hasattr(model, "coattn") and isinstance(model.coattn, nn.Sequential) and len(model.coattn) > 0:
        layer0 = model.coattn[0]
        if hasattr(layer0, "set_attn_mode"):
            layer0.set_attn_mode("self")


def patch_mmp_replace_gene_encoder_with_linear(model: nn.Module, out_dim: int = 256) -> None:
    if not hasattr(model, "sig_networks"):
        return
    try:
        ref_param = next(model.parameters())
        target_device = ref_param.device
        target_dtype = ref_param.dtype
    except StopIteration:
        target_device = torch.device("cpu")
        target_dtype = torch.float32
    nets = []
    for net in model.sig_networks:
        in_dim = None
        for m in net.modules():
            if isinstance(m, nn.Linear):
                in_dim = int(m.in_features)
                break
        if in_dim is None:
            continue
        new_net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        new_net = new_net.to(device=target_device, dtype=target_dtype)
        nets.append(new_net)
    if nets:
        model.sig_networks = nn.ModuleList(nets)


class ScaledMMAttention(nn.Module):
    def __init__(self, base_attn: nn.Module) -> None:
        super().__init__()
        self.base_attn = base_attn
        self.scale_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.base_attn, "forward"):
            raise RuntimeError("Base attention has no forward.")
        out = self.base_attn(*args, **kwargs)
        return out


def patch_mmp_add_scale_after_coattn0(model: nn.Module) -> nn.Parameter | None:
    if not hasattr(model, "coattn") or not isinstance(model.coattn, nn.Sequential) or len(model.coattn) == 0:
        return None

    layer0 = model.coattn[0]
    if not hasattr(layer0, "attn"):
        return None

    base_attn = layer0.attn
    if hasattr(base_attn, "_scaled_replaced"):
        return getattr(base_attn, "scale_logit", None)

    scale_logit = nn.Parameter(torch.tensor(-2.0))
    base_attn.scale_logit = scale_logit
    base_attn._scaled_replaced = True

    old_forward = base_attn.forward

    def new_forward(x: torch.Tensor, mask: torch.Tensor | None = None, return_attn: bool = False) -> Any:
        b, n = x.shape[0], x.shape[1]
        num_pathways = int(getattr(base_attn, "num_pathways", 0))
        if num_pathways <= 0:
            return old_forward(x, mask=mask, return_attn=return_attn)
        h = int(getattr(base_attn, "heads", 1))
        scale = float(getattr(base_attn, "scale", 1.0))
        to_qkv = getattr(base_attn, "to_qkv")
        q, k, v = to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, h, -1).permute(0, 2, 1, 3)
        k = k.view(b, n, h, -1).permute(0, 2, 1, 3)
        v = v.view(b, n, h, -1).permute(0, 2, 1, 3)
        if mask is not None:
            mask_ = mask.view(b, 1, n, 1)
            q = q * mask_
            k = k * mask_
            v = v * mask_
        q = q * scale
        q_path = q[:, :, :num_pathways, :]
        k_path = k[:, :, :num_pathways, :]
        q_h = q[:, :, num_pathways:, :]
        k_h = k[:, :, num_pathways:, :]
        cross_h_logits = torch.einsum("... i d, ... j d -> ... i j", q_h, k_path)
        attn_path_logits = torch.einsum("... i d, ... j d -> ... i j", q_path, k_path)
        cross_p_logits = torch.einsum("... i d, ... j d -> ... i j", q_path, k_h)
        attn_h_logits = torch.einsum("... i d, ... j d -> ... i j", q_h, k_h)
        s = torch.sigmoid(scale_logit)
        mode = getattr(base_attn, "attn_mode", "full")
        pre_softmax_cross_attn_histology = cross_h_logits
        if mode == "full":
            cross_h = (s * cross_h_logits).softmax(dim=-1)
            attn_h_path = torch.cat((cross_h, attn_h_logits), dim=-1).softmax(dim=-1)
            attn_p_h = torch.cat((attn_path_logits, s * cross_p_logits), dim=-1).softmax(dim=-1)
            out_p = attn_p_h @ v
            out_h = attn_h_path @ v
        elif mode == "cross":
            cross_h = (s * cross_h_logits).softmax(dim=-1)
            cross_p = (s * cross_p_logits).softmax(dim=-1)
            out_p = cross_p @ v[:, :, num_pathways:]
            out_h = cross_h @ v[:, :, :num_pathways]
        elif mode == "self":
            attn_h = attn_h_logits.softmax(dim=-1)
            attn_p = attn_path_logits.softmax(dim=-1)
            out_p = attn_p @ v[:, :, :num_pathways]
            out_h = attn_h @ v[:, :, num_pathways:]
        elif mode == "partial":
            cross_h = (s * cross_h_logits).softmax(dim=-1)
            attn_p_h = torch.cat((attn_path_logits, s * cross_p_logits), dim=-1).softmax(dim=-1)
            out_p = attn_p_h @ v
            out_h = cross_h @ v[:, :, :num_pathways]
        else:
            return old_forward(x, mask=mask, return_attn=return_attn)
        out = torch.cat((out_p, out_h), dim=2)
        if getattr(base_attn, "residual", False):
            out = out + base_attn.res_conv(v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        if return_attn:
            return (
                out,
                attn_path_logits.squeeze().detach().cpu(),
                cross_p_logits.squeeze().detach().cpu(),
                pre_softmax_cross_attn_histology.squeeze().detach().cpu(),
            )
        return out

    base_attn.forward = new_forward
    return scale_logit


def load_latent_export(export_pt: Path) -> dict[str, dict[str, np.ndarray]]:
    obj = torch_load_compat(export_pt)
    export: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "test"):
        split_obj = obj.get(split_name)
        if not isinstance(split_obj, dict):
            raise TypeError(f"Missing split dict: {split_name}")
        export[split_name] = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in split_obj.items()}
    return export


def read_ours_predictions_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def read_mmp_predictions_from_dump(test_results_pkl: Path) -> pd.DataFrame:
    obj = load_pkl(test_results_pkl)
    df = pd.DataFrame(
        {
            "sample_id": [str(x) for x in np.asarray(obj["sample_ids"]).tolist()],
            "event_time": np.asarray(obj["all_event_times"]).astype(np.float32),
            "censorship": np.asarray(obj["all_censorships"]).astype(np.float32),
            "pred_risk": np.asarray(obj["all_risk_scores"]).astype(np.float32),
        }
    )
    return df


def bootstrap_mean_diff_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape for paired bootstrap.")
    diff = x - y
    n = int(diff.shape[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmp_src", default="/root/autodl-tmp/_refs/MMP-main/src")
    parser.add_argument("--results_root", default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_plip_luad_256_official_dss/results_5fold_v2pre_453cov")
    parser.add_argument("--export_root", default="/root/autodl-tmp/R2wsp/data_external/main_latent_fair_5fold")
    parser.add_argument("--ours_outputs_root", default="/root/autodl-tmp/mmp-test/R2wsp/outputs")
    parser.add_argument("--ours_experiment", default="")
    parser.add_argument("--ours_summary_json", default="")
    parser.add_argument("--ours_param_count", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir", default="/root/autodl-tmp/R2wsp/data_external/mmp_deep_analysis")
    parser.add_argument("--fit_scale_steps", type=int, default=0)
    parser.add_argument("--fit_scale_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))

    mmp_src = Path(args.mmp_src).resolve()
    import_mmp_src(mmp_src)

    results_root = Path(args.results_root).resolve()
    runs = discover_mmp_fold_runs(results_root)
    if not runs:
        raise RuntimeError(f"No MMP fold runs discovered under: {results_root}")

    export_root = Path(args.export_root).resolve()
    fold_rows: list[dict[str, object]] = []
    attn_rows: list[dict[str, object]] = []
    pred_compare_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []

    all_mmp_fold_test: list[float] = []
    all_ours_fold_test: list[float] = []

    for run in runs:
        config = normalize_mmp_config(resolve_omics_dir(load_mmp_config(run.config_path)))
        loaders = build_mmp_dataloaders_from_config(config)
        omic_sizes = list(getattr(loaders["train"].dataset, "omic_sizes"))
        model = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)

        ckpt_obj = torch.load(run.checkpoint_path, map_location="cpu")
        state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
        model.load_state_dict(state, strict=False)

        params = parameter_breakdown(model)
        params["fold"] = run.fold
        params["run_dir"] = str(run.run_dir)
        param_rows.append({k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in params.items()})

        test_dump = load_pkl(run.test_results_path)
        base_ci = concordance_index(
            np.asarray(test_dump["all_risk_scores"]).astype(np.float32),
            np.asarray(test_dump["all_event_times"]).astype(np.float32),
            np.asarray(test_dump["all_censorships"]).astype(np.float32),
        )
        fold_rows.append({"fold": run.fold, "mmp_test_c_index": float(base_ci), "run_dir": str(run.run_dir)})
        all_mmp_fold_test.append(float(base_ci))

        if "all_omic_attn" in test_dump:
            omic_attn = np.asarray(test_dump["all_omic_attn"]).astype(np.float32)
            cross_attn = np.asarray(test_dump["all_cross_attn"]).astype(np.float32)
            path_attn = np.asarray(test_dump["all_path_attn"]).astype(np.float32)
            stats: dict[str, object] = {"fold": run.fold}
            stats.update(attention_stats("omic_attn", omic_attn, axis=-1))
            stats.update(attention_stats("cross_attn", cross_attn, axis=-1))
            stats.update(attention_stats("path_attn", path_attn, axis=-1))
            stats["omic_attn_shape"] = str(tuple(omic_attn.shape))
            stats["cross_attn_shape"] = str(tuple(cross_attn.shape))
            stats["path_attn_shape"] = str(tuple(path_attn.shape))
            attn_rows.append(stats)

        mmp_pred_df = read_mmp_predictions_from_dump(run.test_results_path)
        mmp_pred_df["fold"] = run.fold

        ours_best = None
        if str(args.ours_experiment).strip():
            ours_root = Path(args.ours_outputs_root).resolve()
            candidates = sorted(ours_root.glob("**/diagnostics/*/test_predictions_seed0.csv"))
            selected = [p for p in candidates if args.ours_experiment in str(p)]
            if selected:
                ours_best = selected[-1]
        if ours_best is not None:
            ours_df = read_ours_predictions_csv(ours_best)
            merged = mmp_pred_df.merge(ours_df[["sample_id", "pred_risk"]].rename(columns={"pred_risk": "ours_pred_risk"}), on="sample_id", how="inner")
            merged = merged.rename(columns={"pred_risk": "mmp_pred_risk"})
            merged["abs_diff"] = (merged["mmp_pred_risk"] - merged["ours_pred_risk"]).abs()
            merged["signed_diff"] = merged["ours_pred_risk"] - merged["mmp_pred_risk"]
            pred_compare_rows.extend(merged.to_dict(orient="records"))

        latent_path = export_root / f"k={run.fold}" / "mmp_latent_export.pt"
        if latent_path.exists():
            export = load_latent_export(latent_path)
            wsi_train = export["train"]["wsi_embedding"].astype(np.float32)
            rna_train = export["train"]["rna_embedding"].astype(np.float32)
            wsi_test = export["test"]["wsi_embedding"].astype(np.float32)
            rna_test = export["test"]["rna_embedding"].astype(np.float32)
            similarity_rows.append({"fold": run.fold, "kind": "wsi", **cosine_similarity_report(wsi_train, wsi_test)})
            similarity_rows.append({"fold": run.fold, "kind": "rna", **cosine_similarity_report(rna_train, rna_test)})
            fusion_train = np.concatenate([wsi_train, rna_train], axis=1).astype(np.float32)
            fusion_test = np.concatenate([wsi_test, rna_test], axis=1).astype(np.float32)
            similarity_rows.append({"fold": run.fold, "kind": "fusion", **cosine_similarity_report(fusion_train, fusion_test)})

        eval_base = evaluate_mmp_model(model, loaders["test"], device=device, loss_fn=str(config.get("loss_fn", "nll")), return_attn=False)
        ablation_rows.append({"fold": run.fold, "ablation": "base", "test_c_index": float(eval_base["c_index"])})

        model_no_wsi = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)
        model_no_wsi.load_state_dict(state, strict=False)
        patch_mmp_disable_cross_attn(model_no_wsi)
        def zero_path(batch: dict[str, Any]) -> dict[str, Any]:
            batch = dict(batch)
            img = batch["img"]
            if isinstance(img, list):
                img = img[0]
            batch["img"] = torch.zeros_like(img)
            return batch

        eval_no_wsi_risk: list[float] = []
        eval_no_wsi_time: list[float] = []
        eval_no_wsi_cens: list[float] = []
        for batch in loaders["test"]:
            batch = zero_path(batch)
            out = extract_mmp_embeddings_and_attn(model_no_wsi, batch, device=device, return_attn=False)
            risk = risk_from_logits(out["logits"], loss_fn=str(config.get("loss_fn", "nll"))).detach().cpu().numpy().reshape(-1)
            eval_no_wsi_risk.append(risk)
            eval_no_wsi_time.append(batch["survival_time"].detach().cpu().numpy().reshape(-1))
            eval_no_wsi_cens.append(batch["censorship"].detach().cpu().numpy().reshape(-1))
        risk_all = np.concatenate(eval_no_wsi_risk).astype(np.float32)
        times_all = np.concatenate(eval_no_wsi_time).astype(np.float32)
        cens_all = np.concatenate(eval_no_wsi_cens).astype(np.float32)
        ablation_rows.append({"fold": run.fold, "ablation": "no_wsi_infer", "test_c_index": concordance_index(risk_all, times_all, cens_all)})

        model_no_rna = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)
        model_no_rna.load_state_dict(state, strict=False)
        patch_mmp_disable_cross_attn(model_no_rna)
        eval_no_rna_risk: list[np.ndarray] = []
        eval_no_rna_time: list[np.ndarray] = []
        eval_no_rna_cens: list[np.ndarray] = []
        for batch in loaders["test"]:
            batch = dict(batch)
            omics = batch["omics"]
            batch["omics"] = [torch.zeros_like(x) for x in omics]
            out = extract_mmp_embeddings_and_attn(model_no_rna, batch, device=device, return_attn=False)
            risk = risk_from_logits(out["logits"], loss_fn=str(config.get("loss_fn", "nll"))).detach().cpu().numpy().reshape(-1)
            eval_no_rna_risk.append(risk)
            eval_no_rna_time.append(batch["survival_time"].detach().cpu().numpy().reshape(-1))
            eval_no_rna_cens.append(batch["censorship"].detach().cpu().numpy().reshape(-1))
        risk_all = np.concatenate(eval_no_rna_risk).astype(np.float32)
        times_all = np.concatenate(eval_no_rna_time).astype(np.float32)
        cens_all = np.concatenate(eval_no_rna_cens).astype(np.float32)
        ablation_rows.append({"fold": run.fold, "ablation": "no_rna_infer", "test_c_index": concordance_index(risk_all, times_all, cens_all)})

        model_selfattn = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)
        model_selfattn.load_state_dict(state, strict=False)
        patch_mmp_disable_cross_attn(model_selfattn)
        eval_self = evaluate_mmp_model(model_selfattn, loaders["test"], device=device, loss_fn=str(config.get("loss_fn", "nll")), return_attn=False)
        ablation_rows.append({"fold": run.fold, "ablation": "no_crossattn_infer", "test_c_index": float(eval_self["c_index"])})

        model_gene_linear = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)
        model_gene_linear.load_state_dict(state, strict=False)
        patch_mmp_replace_gene_encoder_with_linear(model_gene_linear, out_dim=256)
        eval_gene_linear = evaluate_mmp_model(model_gene_linear, loaders["test"], device=device, loss_fn=str(config.get("loss_fn", "nll")), return_attn=False)
        ablation_rows.append({"fold": run.fold, "ablation": "gene_encoder_linear_infer", "test_c_index": float(eval_gene_linear["c_index"])})

        if int(args.fit_scale_steps) > 0:
            model_scale = build_mmp_model_from_config(config, omic_sizes=omic_sizes).to(device)
            model_scale.load_state_dict(state, strict=False)
            scale_logit = patch_mmp_add_scale_after_coattn0(model_scale)
            if scale_logit is not None:
                for p in model_scale.parameters():
                    p.requires_grad_(False)
                scale_logit.requires_grad_(True)
                opt = torch.optim.Adam([scale_logit], lr=float(args.fit_scale_lr))
                loss_fn_name = str(config.get("loss_fn", "nll"))
                for _step in range(int(args.fit_scale_steps)):
                    for batch in loaders.get("train", loaders["test"]):
                        out = extract_mmp_embeddings_and_attn(model_scale, batch, device=device, return_attn=False)
                        logits = out["logits"]
                        if loss_fn_name == "nll":
                            label = batch["label"].to(device)
                            censorship = batch["censorship"].to(device)
                            loss = torch.nn.functional.cross_entropy(logits, label.long().view(-1), reduction="mean")
                            loss = loss + 0.0 * censorship.mean()
                        else:
                            loss = logits.mean() * 0.0
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        break
                ablation_rows.append(
                    {
                        "fold": run.fold,
                        "ablation": "fitted_scale_logit",
                        "value": float(scale_logit.detach().cpu().item()),
                        "scale": float(torch.sigmoid(scale_logit.detach()).cpu().item()),
                    }
                )

    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    write_csv(out_dir / "mmp_fold_test_cindex.csv", fold_df.to_dict(orient="records"), ["fold", "mmp_test_c_index", "run_dir"])
    if attn_rows:
        attn_df = pd.DataFrame(attn_rows).sort_values("fold")
        attn_df.to_csv(out_dir / "mmp_attention_stats.csv", index=False)
    if param_rows:
        param_df = pd.DataFrame(param_rows).sort_values("fold")
        param_df.to_csv(out_dir / "mmp_parameter_breakdown.csv", index=False)
    if ablation_rows:
        abl_df = pd.DataFrame(ablation_rows).sort_values(["fold", "ablation"])
        abl_df.to_csv(out_dir / "mmp_inference_ablations.csv", index=False)
    if similarity_rows:
        sim_df = pd.DataFrame(similarity_rows).sort_values(["fold", "kind"])
        sim_df.to_csv(out_dir / "latent_similarity_report.csv", index=False)
    if pred_compare_rows:
        pred_df = pd.DataFrame(pred_compare_rows)
        pred_df.to_csv(out_dir / "per_sample_risk_compare.csv", index=False)

    overall: dict[str, object] = {
        "results_root": str(results_root),
        "export_root": str(export_root),
        "n_folds": int(len(runs)),
        "mmp_mean_test_c_index": float(np.mean(all_mmp_fold_test)) if all_mmp_fold_test else float("nan"),
        "mmp_std_test_c_index": float(np.std(all_mmp_fold_test)) if all_mmp_fold_test else float("nan"),
    }
    save_json(out_dir / "summary.json", overall)

    comparison_rows: list[dict[str, object]] = []
    comparison_rows.append(
        {
            "model": "mmp__original_end2end",
            "param_total": int(param_rows[0]["total"]) if param_rows else 0,
            "mean_test_c_index": float(overall["mmp_mean_test_c_index"]),
            "std_test_c_index": float(overall["mmp_std_test_c_index"]),
        }
    )
    if str(args.ours_summary_json).strip():
        ours_path = Path(args.ours_summary_json).resolve()
        if ours_path.exists():
            ours_obj = json.loads(ours_path.read_text(encoding="utf-8"))
            best = ours_obj.get("best_experiment") or {}
            comparison_rows.append(
                {
                    "model": "ours_fusion",
                    "param_total": int(args.ours_param_count) if int(args.ours_param_count) > 0 else 0,
                    "mean_test_c_index": float(best.get("mean_test_c_index", float("nan"))),
                    "std_test_c_index": float(best.get("std_test_c_index", float("nan"))),
                }
            )
    if comparison_rows:
        write_csv(out_dir / "comparison_summary.csv", comparison_rows, ["model", "param_total", "mean_test_c_index", "std_test_c_index"])

    if all_ours_fold_test and len(all_ours_fold_test) == len(all_mmp_fold_test):
        boot = bootstrap_mean_diff_ci(np.asarray(all_ours_fold_test), np.asarray(all_mmp_fold_test), n_boot=int(args.n_boot), seed=int(args.seed))
        save_json(out_dir / "paired_bootstrap_significance.json", boot)

    if bool(args.plot):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.plot(fold_df["fold"].tolist(), fold_df["mmp_test_c_index"].tolist(), marker="o")
            ax.set_xlabel("fold")
            ax.set_ylabel("test_c_index")
            ax.set_title("MMP fold test c-index")
            fig.tight_layout()
            fig.savefig(out_dir / "mmp_fold_test_cindex.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
