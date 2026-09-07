from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_split_dir(root: Path, *, cohort: str, stage: int, fold: int) -> Path:
    base = root / "data" / "splits" / f"censored_stage_protocol_{cohort}"
    if stage == 1:
        return base / "phase1_outer5" / f"fold_{fold}"
    return base / "phase2_independent5" / f"fold_{fold}" / "eval_with_censored"


def _build_split_dir_stage1_all(root: Path, *, cohort: str, fold: int) -> Path:
    base = root / "data" / "splits" / f"censored_stage_protocol_{cohort}"
    return base / "phase2_independent5" / f"fold_{fold}" / "eval_with_censored"


def _build_out_dir(root: Path, *, cohort: str, seed: int, fold: int, stage: int) -> Path:
    stage_name = "stage1" if stage == 1 else "stage2"
    return root / "outputs" / "sota_uni1024_hallmark_omics" / cohort / f"seed{seed}" / f"fold{fold}" / stage_name


def _build_out_dir_with_exp(root: Path, *, exp_name: str, cohort: str, seed: int, fold: int, stage: int) -> Path:
    stage_name = "stage1" if stage == 1 else "stage2"
    return root / "outputs" / str(exp_name) / cohort / f"seed{seed}" / f"fold{fold}" / stage_name


def _build_pretrain_ckpt(root: Path, *, exp_name: str, cohort: str, seed: int, fold: int) -> Path:
    return _build_out_dir_with_exp(root, exp_name=exp_name, cohort=cohort, seed=seed, fold=fold, stage=1) / "best.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--exp_name", default="sota_uni1024_hallmark_omics")
    p.add_argument("--pretrain_exp_name", default=None, help="stage=2 时指定复用哪个 exp_name 的 stage1/best.pt；默认=使用 --exp_name。")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--lr_min", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--selection_min_epochs", type=int, default=8)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max_tiles", type=int, default=256)
    p.add_argument("--window", default="param_safe", choices=["param_safe", "param_best_mix", "onek", "twok", "fixed_off"])
    p.add_argument(
        "--stage1_split",
        default="phase1_uncensored",
        choices=["phase1_uncensored", "phase2_all"],
    )
    args = p.parse_args()

    root = _resolve_root()
    cohort = str(args.cohort).upper()
    stage = int(args.stage)
    seed = int(args.seed)
    fold = int(args.fold)
    exp_name = str(args.exp_name)
    pretrain_exp_name = exp_name if args.pretrain_exp_name is None else str(args.pretrain_exp_name)

    if stage == 1 and str(args.stage1_split) == "phase2_all":
        split_dir = _build_split_dir_stage1_all(root, cohort=cohort, fold=fold)
    else:
        split_dir = _build_split_dir(root, cohort=cohort, stage=stage, fold=fold)
    out_dir = _build_out_dir_with_exp(root, exp_name=exp_name, cohort=cohort, seed=seed, fold=fold, stage=stage)

    cmd: list[str] = [
        sys.executable,
        str(root / "scripts" / "train_censored_stage_survival.py"),
        "--stage",
        str(stage),
        "--cohort_hint",
        cohort,
        "--wsi_feature_source",
        "uni1024",
        "--rna_mode",
        "omics",
        "--rna_gene_sets_csv",
        str(root / "data" / "raw_rna" / "metadata" / "hallmarks_signatures.csv"),
        "--gene_annotation_gtf",
        "/root/autodl-tmp/gencode.v22.annotation.gtf.gz",
        "--split_dir",
        str(split_dir),
        "--out_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--batch_size",
        str(int(args.batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(args.lr)),
        "--lr_scheduler",
        str(args.lr_scheduler),
        "--lr_min",
        str(float(args.lr_min)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--selection_min_epochs",
        str(int(args.selection_min_epochs)),
        "--early_stop_patience",
        str(int(args.early_stop_patience)),
        "--hidden_dim",
        str(int(args.hidden_dim)),
        "--dropout",
        str(float(args.dropout)),
        "--max_tiles",
        str(int(args.max_tiles)),
        "--use_multi_slide",
        "--multi_slide_mode",
        "slide_attn_case_attn",
        "--multi_slide_tile_budget_mode",
        "case_shared",
    ]

    if stage == 2:
        ckpt = _build_pretrain_ckpt(root, exp_name=pretrain_exp_name, cohort=cohort, seed=seed, fold=fold)
        if not ckpt.exists():
            raise FileNotFoundError(f"stage1 best.pt not found: {ckpt}")
        cmd += ["--pretrain_checkpoint", str(ckpt)]

    if stage == 2:
        if str(args.window) == "fixed_off":
            cmd += [
                "--loss_window_lower",
                "0",
                "--loss_window_upper",
                "1",
                "--loss_window_A",
                "1.0",
                "--loss_window_eps",
                "1.0",
                "--loss_window_policy",
                "param",
                "--loss_window_center_mode",
                "fixed",
                "--loss_window_width_mode",
                "fixed",
                "--loss_window_transition",
                "0.0",
            ]
        elif str(args.window) == "param_best_mix":
            cmd += [
                "--loss_window_policy",
                "param",
                "--loss_window_metric",
                "val_c_index_ema",
                "--loss_window_center_mode",
                "best_mix",
                "--loss_window_best_mix_alpha",
                "0.5",
                "--loss_window_width_mode",
                "quantile",
                "--loss_window_width_quantile",
                "0.8",
                "--loss_window_transition",
                "0.25",
                "--loss_window_width_min",
                "0.06",
                "--loss_window_eps",
                "0.05",
            ]
        elif str(args.window) == "onek":
            cmd += [
                "--loss_window_policy",
                "onek",
                "--loss_window_K",
                "0.25",
                "--loss_window_eps",
                "0.05",
            ]
        elif str(args.window) == "twok":
            cmd += [
                "--loss_window_policy",
                "twok",
                "--loss_window_K_left",
                "0.35",
                "--loss_window_K_right",
                "0.20",
                "--loss_window_eps",
                "0.05",
            ]
        else:
            cmd += [
                "--loss_window_policy",
                "param",
                "--loss_window_metric",
                "val_c_index_ema",
                "--loss_window_center_mode",
                "ema",
                "--loss_window_width_mode",
                "quantile",
                "--loss_window_width_quantile",
                "0.8",
                "--loss_window_transition",
                "0.25",
                "--loss_window_width_min",
                "0.06",
                "--loss_window_eps",
                "0.05",
            ]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
