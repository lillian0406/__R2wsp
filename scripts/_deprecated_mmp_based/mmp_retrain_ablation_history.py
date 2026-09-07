from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def import_mmp_src(mmp_src: Path) -> None:
    sys.path.insert(0, str(mmp_src))
    sys.path.insert(0, str(mmp_src / "training"))
    sys.path.insert(0, str(mmp_src.parent))
    if (mmp_src / "data_csvs" / "rna" / "metadata").is_dir():
        os.chdir(mmp_src)


def import_main_survival_module() -> Any:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        return importlib.import_module("training.main_survival")
    finally:
        sys.argv = old_argv


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


def patch_disable_cross_attn(model: Any) -> None:
    if hasattr(model, "coattn") and hasattr(model.coattn, "__getitem__") and len(model.coattn) > 0:
        layer0 = model.coattn[0]
        if hasattr(layer0, "set_attn_mode"):
            layer0.set_attn_mode("self")


def patch_replace_gene_encoder_with_linear(model: Any, out_dim: int = 256) -> None:
    import torch.nn as nn

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


def patch_add_scale_after_coattn0(model: Any) -> torch.nn.Parameter | None:
    import torch.nn as nn

    if not hasattr(model, "coattn") or len(model.coattn) == 0:
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


@dataclass
class AblationSpec:
    name: str
    drop_wsi: bool = False
    drop_rna: bool = False
    disable_cross_attn: bool = False
    gene_encoder_linear: bool = False
    scale_after_attn: bool = False


def make_ablation_specs() -> list[AblationSpec]:
    return [
        AblationSpec(name="base"),
        AblationSpec(name="no_wsi", drop_wsi=True),
        AblationSpec(name="no_rna", drop_rna=True),
        AblationSpec(name="no_cross_attn", disable_cross_attn=True),
        AblationSpec(name="gene_encoder_linear", gene_encoder_linear=True),
        AblationSpec(name="cross_attn_scale", scale_after_attn=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmp_src", default="/root/autodl-tmp/_refs/MMP-main/src")
    parser.add_argument("--split_dir_root", default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_plip_luad_256_official_dss/splits/survival")
    parser.add_argument("--data_source", default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_plip_luad_256_official_dss/histology/extracted_mag20x_patch256_fp/plip_luad_256/feats_pt")
    parser.add_argument("--omics_dir_root", default="/root/autodl-tmp/_refs/MMP-main/src/data_csvs/rna")
    parser.add_argument("--type_of_path", default="hallmarks")
    parser.add_argument("--omics_modality", default="pathway")
    parser.add_argument("--task", default="LUAD_survival")
    parser.add_argument("--target_col", default="dss_survival_days")
    parser.add_argument("--model_histo_type", default="MIL")
    parser.add_argument("--model_histo_config", default="MIL_default")
    parser.add_argument("--model_mm_type", default="survpath")
    parser.add_argument("--histo_agg", default="mean")
    parser.add_argument("--append_embed", default="none")
    parser.add_argument("--net_indiv", action="store_true")
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--bag_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--opt", default="adamW", choices=["adamW", "sgd", "RAdam"])
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--loss_fn", default="nll", choices=["nll", "cox", "rank"])
    parser.add_argument("--n_label_bins", type=int, default=4)
    parser.add_argument("--nll_alpha", type=float, default=0.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_root", default="/root/autodl-tmp/R2wsp/data_external/mmp_retrain_ablations")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dumps", action="store_true")
    parser.add_argument("--save_attn", action="store_true")
    parser.add_argument("--run_only", default="")
    args = parser.parse_args()

    device = resolve_device(str(args.device))
    import_mmp_src(Path(args.mmp_src).resolve())

    main_survival = import_main_survival_module()
    if hasattr(main_survival, "args"):
        setattr(main_survival.args, "loss_fn", str(args.loss_fn))
    build_datasets = main_survival.build_datasets
    from training.trainer import train_loop_survival, validate_survival
    from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
    from utils.utils import get_optim, get_lr_scheduler, seed_torch, read_splits
    from mil_models.model_factory import create_multimodal_survival_model

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    folds = [int(x.strip()) for x in str(args.folds).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    ablations = make_ablation_specs()
    if str(args.run_only).strip():
        allow = {x.strip() for x in str(args.run_only).split(",") if x.strip()}
        ablations = [a for a in ablations if a.name in allow]

    for fold in folds:
        split_dir = Path(args.split_dir_root).resolve() / f"TCGA_LUAD_overall_survival_k={fold}"
        cancer_type = "LUAD"
        omics_dir = Path(args.omics_dir_root).resolve() / str(args.type_of_path) / cancer_type
        for seed in seeds:
            seed_torch(seed)
            for ab in ablations:
                run_dir = out_root / f"k={fold}" / f"seed={seed}" / ab.name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "histories").mkdir(parents=True, exist_ok=True)
                (run_dir / "dumps").mkdir(parents=True, exist_ok=True)

                config = {
                    "data_source": [str(Path(args.data_source).resolve())],
                    "omics_dir": str(omics_dir),
                    "split_dir": str(split_dir),
                    "split_names": "train,test",
                    "task": str(args.task),
                    "target_col": str(args.target_col),
                    "model_histo_type": str(args.model_histo_type),
                    "model_histo_config": str(args.model_histo_config),
                    "model_mm_type": str(args.model_mm_type),
                    "histo_agg": str(args.histo_agg),
                    "append_embed": str(args.append_embed),
                    "net_indiv": bool(args.net_indiv),
                    "in_dim": int(args.in_dim),
                    "bag_size": int(args.bag_size),
                    "train_bag_size": int(args.bag_size),
                    "val_bag_size": int(args.bag_size),
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                    "max_epochs": int(args.max_epochs),
                    "lr": float(args.lr),
                    "wd": float(args.wd),
                    "opt": str(args.opt),
                    "lr_scheduler": str(args.lr_scheduler),
                    "warmup_steps": int(args.warmup_steps),
                    "warmup_epochs": int(args.warmup_epochs),
                    "loss_fn": str(args.loss_fn),
                    "n_label_bins": int(args.n_label_bins),
                    "nll_alpha": float(args.nll_alpha),
                    "accum_steps": int(args.accum_steps),
                    "print_every": int(args.print_every),
                    "seed": int(seed),
                    "early_stopping": 0,
                    "es_min_epochs": 3,
                    "es_patience": 5,
                    "es_metric": "loss",
                    "omics_modality": str(args.omics_modality),
                    "type_of_path": str(args.type_of_path),
                    "feat_dim": int(args.in_dim),
                    "results_dir": str(run_dir),
                }
                save_json(run_dir / "config.json", config)

                class ArgsObj:
                    def __init__(self, d: dict[str, Any]) -> None:
                        self.__dict__.update(d)

                    def __contains__(self, key: str) -> bool:
                        return hasattr(self, key)

                args_obj = ArgsObj(config)
                csv_splits = read_splits(args_obj)
                train_kwargs = dict(
                    data_source=args_obj.data_source,
                    survival_time_col=args_obj.target_col,
                    censorship_col=args_obj.target_col.split("_")[0] + "_censorship",
                    n_label_bins=args_obj.n_label_bins,
                    label_bins=None,
                    bag_size=args_obj.train_bag_size,
                    shuffle=True,
                    omics_dir=args_obj.omics_dir,
                    omics_modality=args_obj.omics_modality,
                )
                val_kwargs = dict(
                    data_source=args_obj.data_source,
                    survival_time_col=args_obj.target_col,
                    censorship_col=args_obj.target_col.split("_")[0] + "_censorship",
                    n_label_bins=args_obj.n_label_bins,
                    label_bins=None,
                    bag_size=args_obj.val_bag_size,
                    shuffle=False,
                    omics_dir=args_obj.omics_dir,
                    omics_modality=args_obj.omics_modality,
                )
                datasets = build_datasets(csv_splits, model_type=args_obj.model_histo_type, batch_size=args_obj.batch_size, num_workers=args_obj.num_workers, train_kwargs=train_kwargs, val_kwargs=val_kwargs)
                omic_sizes = list(getattr(datasets["train"].dataset, "omic_sizes"))

                if args_obj.loss_fn == "nll":
                    loss_fn = NLLSurvLoss(alpha=args_obj.nll_alpha)
                elif args_obj.loss_fn == "cox":
                    loss_fn = CoxLoss()
                else:
                    loss_fn = SurvRankingLoss()

                model = create_multimodal_survival_model(args_obj, omic_sizes=omic_sizes).to(device)
                if ab.disable_cross_attn:
                    patch_disable_cross_attn(model)
                if ab.gene_encoder_linear:
                    patch_replace_gene_encoder_with_linear(model, out_dim=256)
                scale_logit = patch_add_scale_after_coattn0(model) if ab.scale_after_attn else None

                optimizer = get_optim(model=model, args=args_obj)
                lr_scheduler = get_lr_scheduler(args_obj, optimizer, datasets["train"])

                history_rows: list[dict[str, object]] = []
                for epoch in range(int(args_obj.max_epochs)):
                    if ab.drop_wsi or ab.drop_rna:
                        model.train()
                        for batch_idx, batch in enumerate(datasets["train"]):
                            if ab.drop_wsi:
                                img = batch["img"]
                                if isinstance(img, list):
                                    img = img[0]
                                batch["img"] = torch.zeros_like(img)
                            if ab.drop_rna:
                                batch["omics"] = [torch.zeros_like(x) for x in batch["omics"]]
                            data = batch["img"]
                            if isinstance(data, list):
                                data = data[0]
                            data = data.to(device)
                            label = batch["label"].to(device)
                            censorship = batch["censorship"].to(device)
                            omics = [x.to(device) for x in batch["omics"]]
                            out, _log = model(data, omics, label=label, censorship=censorship, loss_fn=loss_fn)
                            loss = out["loss"]
                            if loss is None:
                                continue
                            (loss / float(args_obj.accum_steps)).backward()
                            if (batch_idx + 1) % int(args_obj.accum_steps) == 0:
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()
                    else:
                        _ = train_loop_survival(model, datasets["train"], optimizer, lr_scheduler, loss_fn, print_every=int(args_obj.print_every), accum_steps=int(args_obj.accum_steps))

                    if int(args.eval_every) > 0 and ((epoch + 1) % int(args.eval_every) == 0 or epoch == int(args_obj.max_epochs) - 1):
                        train_res, _ = validate_survival(model, datasets["train"], loss_fn, print_every=int(args_obj.print_every), dump_results=False, return_attn=False, verbose=False)
                        test_res, test_dump = validate_survival(model, datasets["test"], loss_fn, print_every=int(args_obj.print_every), dump_results=bool(args.save_dumps), return_attn=bool(args.save_attn), verbose=False)
                        row: dict[str, object] = {
                            "fold": fold,
                            "seed": seed,
                            "ablation": ab.name,
                            "epoch": int(epoch),
                            "train_loss": float(train_res.get("loss", float("nan"))),
                            "train_c_index": float(train_res.get("c_index", float("nan"))),
                            "test_loss": float(test_res.get("loss", float("nan"))),
                            "test_c_index": float(test_res.get("c_index", float("nan"))),
                        }
                        if scale_logit is not None:
                            row["scale_logit"] = float(scale_logit.detach().cpu().item())
                            row["scale"] = float(torch.sigmoid(scale_logit.detach()).cpu().item())
                        history_rows.append(row)
                        if bool(args.save_dumps):
                            dump_path = run_dir / "dumps" / f"test_epoch{epoch}.pkl"
                            with dump_path.open("wb") as f:
                                import pickle

                                pickle.dump(test_dump, f)
                write_csv(run_dir / "histories" / "history.csv", history_rows)

    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()
