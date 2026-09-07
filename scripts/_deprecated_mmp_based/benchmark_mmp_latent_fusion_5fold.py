from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_main_project_gated_merge import (  # noqa: E402
    GatedFusionRisk,
    apply_standardize,
    concordance_index,
    load_export,
    neg_partial_log_likelihood,
    permute_survival_labels,
    sample_overlap_report,
    save_json,
    set_seed,
    split_train_val_indices,
    standardize,
    write_csv,
)
from experiment_main_project_geometry_merge import (  # noqa: E402
    FusionBaselineTinyGeometryRisk,
    FusionConcatBaselineRisk,
    FusionGeometryRisk,
    top_geometry_correlations,
)


@dataclass
class RunResult:
    fold: int
    experiment: str
    seed: int
    best_epoch: int
    best_val_c_index: float
    train_c_index: float
    val_c_index: float
    test_c_index: float


def parse_folds(value: str) -> list[int]:
    folds = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not folds:
        raise ValueError("at least one fold is required")
    return folds


def parse_list(value: str) -> list[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        raise ValueError("at least one item is required")
    return items


def resolve_export_root(export_root: str | None, manifest_path: str | None) -> tuple[Path, list[int]]:
    manifest_folds: list[int] | None = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        export_root_resolved = Path(manifest["export_root"]).resolve()
        manifest_folds = [int(x) for x in manifest.get("folds", [])]
    elif export_root:
        export_root_resolved = Path(export_root).resolve()
    else:
        raise ValueError("either --export-root or --manifest-path is required")
    return export_root_resolved, (manifest_folds or [])


def make_model_factory(
    *,
    experiment: str,
    wsi_dim: int,
    rna_dim: int,
    hidden_dim: int,
    dropout: float,
    num_points: int,
    tiny_num_points: int,
    tiny_compact_dim: int,
    tiny_sidecar_hidden_dim: int,
    tiny_residual_scale_init: float,
) -> callable:
    if experiment == "fusion__baseline":
        return lambda: FusionConcatBaselineRisk(
            wsi_dim=wsi_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if experiment == "fusion__gated":
        return lambda: GatedFusionRisk(
            wsi_dim=wsi_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if experiment == "fusion__geometry":
        return lambda: FusionGeometryRisk(
            wsi_dim=wsi_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            num_points=num_points,
            dropout=dropout,
        )
    if experiment == "fusion__baseline_tiny_geometry":
        return lambda: FusionBaselineTinyGeometryRisk(
            wsi_dim=wsi_dim,
            rna_dim=rna_dim,
            hidden_dim=hidden_dim,
            num_points=tiny_num_points,
            dropout=dropout,
            compact_dim=tiny_compact_dim,
            sidecar_hidden_dim=tiny_sidecar_hidden_dim,
            residual_scale_init=tiny_residual_scale_init,
        )
    raise ValueError(f"unsupported experiment: {experiment}")


def summarize_original_mmp_reference(
    *,
    fold: int,
    train_obj: dict[str, np.ndarray],
    test_obj: dict[str, np.ndarray],
) -> dict[str, object]:
    required = ["risk_scores", "event_times", "censorships"]
    for key in required:
        if key not in train_obj or key not in test_obj:
            raise KeyError(f"missing required export key {key} for mmp__original in fold {fold}")
    train_pred = train_obj["risk_scores"].astype(np.float32).reshape(-1)
    test_pred = test_obj["risk_scores"].astype(np.float32).reshape(-1)
    train_times = train_obj["event_times"].astype(np.float32)
    train_censorships = train_obj["censorships"].astype(np.float32)
    test_times = test_obj["event_times"].astype(np.float32)
    test_censorships = test_obj["censorships"].astype(np.float32)
    return {
        "fold": fold,
        "experiment": "mmp__original",
        "num_seeds": 1,
        "mean_val_c_index": float("nan"),
        "std_val_c_index": float("nan"),
        "mean_test_c_index": concordance_index(test_pred, test_times, test_censorships),
        "std_test_c_index": 0.0,
        "mean_train_c_index": concordance_index(train_pred, train_times, train_censorships),
    }


def finite_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def build_survival_strata(times: np.ndarray, censorships: np.ndarray, num_time_bins: int) -> np.ndarray:
    times = times.astype(np.float64)
    censorships = censorships.astype(np.int64)
    if len(times) == 0:
        return np.empty((0,), dtype=np.int64)

    if num_time_bins <= 1:
        return censorships.copy()

    unique_times = np.unique(times)
    if unique_times.size <= 1:
        return censorships.copy()

    quantiles = np.linspace(0.0, 1.0, num_time_bins + 1)[1:-1]
    boundaries = np.unique(np.quantile(times, quantiles))
    if boundaries.size == 0:
        time_bins = np.zeros(len(times), dtype=np.int64)
    else:
        time_bins = np.digitize(times, boundaries, right=False).astype(np.int64)
    return censorships * max(1, num_time_bins) + time_bins


def split_train_val_indices_survival(
    *,
    n: int,
    seed: int,
    val_frac: float,
    times: np.ndarray,
    censorships: np.ndarray,
    split_mode: str,
    num_time_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "random":
        return split_train_val_indices(n, seed, val_frac)

    rng = np.random.default_rng(seed)
    strata = build_survival_strata(times, censorships, num_time_bins=num_time_bins)
    val_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []

    for stratum in np.unique(strata):
        idx = np.flatnonzero(strata == stratum)
        rng.shuffle(idx)
        if idx.size <= 1:
            fit_parts.append(np.sort(idx))
            continue
        val_size = max(1, int(round(idx.size * val_frac)))
        if val_size >= idx.size:
            val_size = idx.size - 1
        val_parts.append(np.sort(idx[:val_size]))
        fit_parts.append(np.sort(idx[val_size:]))

    if not val_parts:
        return split_train_val_indices(n, seed, val_frac)

    fit_idx = np.sort(np.concatenate(fit_parts)) if fit_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_parts))
    if fit_idx.size == 0 or val_idx.size == 0:
        return split_train_val_indices(n, seed, val_frac)
    return fit_idx, val_idx


def run_one_seed_pair(
    *,
    fold: int,
    experiment: str,
    seed: int,
    wsi_train: np.ndarray,
    rna_train: np.ndarray,
    train_times: np.ndarray,
    train_censorships: np.ndarray,
    wsi_test: np.ndarray,
    rna_test: np.ndarray,
    test_times: np.ndarray,
    test_censorships: np.ndarray,
    model_factory: callable,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    split_mode: str,
    num_time_bins: int,
    selection_metric: str,
    min_epochs: int,
    early_stop_patience: int,
    out_dir: Path,
) -> tuple[RunResult, dict[str, object]]:
    set_seed(seed)
    fit_idx, val_idx = split_train_val_indices_survival(
        n=wsi_train.shape[0],
        seed=seed,
        val_frac=val_frac,
        times=train_times,
        censorships=train_censorships,
        split_mode=split_mode,
        num_time_bins=num_time_bins,
    )
    wsi_fit, wsi_val, wsi_test_std, wsi_mean, wsi_std = standardize(wsi_train[fit_idx], wsi_train[val_idx], wsi_test)
    rna_fit, rna_val, rna_test_std, rna_mean, rna_std = standardize(rna_train[fit_idx], rna_train[val_idx], rna_test)

    fit_times = train_times[fit_idx].astype(np.float32)
    fit_censorships = train_censorships[fit_idx].astype(np.float32)
    val_times = train_times[val_idx].astype(np.float32)
    val_censorships = train_censorships[val_idx].astype(np.float32)
    fit_events = 1.0 - fit_censorships
    val_events = 1.0 - val_censorships

    device = torch.device("cpu")
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    wsi_fit_t = torch.from_numpy(wsi_fit).to(device)
    rna_fit_t = torch.from_numpy(rna_fit).to(device)
    wsi_val_t = torch.from_numpy(wsi_val).to(device)
    rna_val_t = torch.from_numpy(rna_val).to(device)
    wsi_test_t = torch.from_numpy(wsi_test_std).to(device)
    rna_test_t = torch.from_numpy(rna_test_std).to(device)
    fit_times_t = torch.from_numpy(fit_times).to(device)
    fit_events_t = torch.from_numpy(fit_events.astype(np.float32)).to(device)
    val_times_t = torch.from_numpy(val_times).to(device)
    val_events_t = torch.from_numpy(val_events.astype(np.float32)).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | int] | None = None
    best_selection_score: float | None = None
    epochs_since_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        fit_risk = model(wsi_fit_t, rna_fit_t)[0]
        loss = neg_partial_log_likelihood(fit_risk, fit_times_t, fit_events_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            fit_pred = model(wsi_fit_t, rna_fit_t)[0].cpu().numpy()
            val_risk = model(wsi_val_t, rna_val_t)[0]
            val_pred = val_risk.cpu().numpy()
            test_pred = model(wsi_test_t, rna_test_t)[0].cpu().numpy()
            val_loss = neg_partial_log_likelihood(val_risk, val_times_t, val_events_t)

        metrics = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "val_loss": float(val_loss.item()),
            "train_c_index": concordance_index(fit_pred, fit_times, fit_censorships),
            "val_c_index": concordance_index(val_pred, val_times, val_censorships),
            "test_c_index": concordance_index(test_pred, test_times, test_censorships),
        }
        history.append(metrics)
        if selection_metric == "val_loss":
            current_score = -float(metrics["val_loss"])
        elif selection_metric == "val_c_index":
            current_score = float(metrics["val_c_index"])
        else:
            raise ValueError(f"unsupported selection_metric: {selection_metric}")

        improved = best_selection_score is None or current_score > best_selection_score + 1e-8
        if improved:
            best_selection_score = current_score
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch >= int(min_epochs) and epochs_since_improvement >= int(early_stop_patience):
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_wsi_all = apply_standardize(wsi_train, wsi_mean, wsi_std)
        train_rna_all = apply_standardize(rna_train, rna_mean, rna_std)
        train_pred = model(
            torch.from_numpy(train_wsi_all).to(device),
            torch.from_numpy(train_rna_all).to(device),
        )[0].cpu().numpy()
        test_pred, aux = model(wsi_test_t, rna_test_t)
        test_pred_np = test_pred.cpu().numpy()

    aux_np: dict[str, np.ndarray] = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            aux_np[key] = value.detach().cpu().numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "histories" / f"k{fold}__{experiment}__seed{seed}.csv",
        history,
        ["epoch", "loss", "val_loss", "train_c_index", "val_c_index", "test_c_index"],
    )

    return (
        RunResult(
            fold=fold,
            experiment=experiment,
            seed=seed,
            best_epoch=int(best_metrics["epoch"]),
            best_val_c_index=float(best_metrics["val_c_index"]),
            train_c_index=concordance_index(train_pred, train_times, train_censorships),
            val_c_index=float(best_metrics["val_c_index"]),
            test_c_index=concordance_index(test_pred_np, test_times, test_censorships),
        ),
        {
            "test_pred": test_pred_np,
            "aux": aux_np,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run a fair cross-fold latent fusion benchmark from matched MMP exports.")
    p.add_argument("--export-root", default=None, help="Directory that contains k=<fold>/mmp_latent_export.pt.")
    p.add_argument("--manifest-path", default=None, help="Manifest produced by export_mmp_latents_5fold.py.")
    p.add_argument("--folds", default="", help="Optional fold override such as 0,1,2,3,4.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--experiments", default="fusion__baseline,fusion__geometry")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num-points", type=int, default=24)
    p.add_argument("--tiny-num-points", type=int, default=8)
    p.add_argument("--tiny-compact-dim", type=int, default=32)
    p.add_argument("--tiny-sidecar-hidden-dim", type=int, default=8)
    p.add_argument("--tiny-residual-scale-init", type=float, default=0.05)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    p.add_argument("--time-bins", type=int, default=4)
    p.add_argument("--selection-metric", choices=["val_loss", "val_c_index"], default="val_loss")
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--run-permutation-check", action="store_true")
    p.add_argument("--perm-epochs", type=int, default=80)
    args = p.parse_args()

    export_root, manifest_folds = resolve_export_root(args.export_root, args.manifest_path)
    folds = parse_folds(args.folds) if str(args.folds).strip() else manifest_folds
    if not folds:
        fold_dirs = sorted(export_root.glob("k=*"))
        folds = [int(path.name.split("=")[1]) for path in fold_dirs]
    if not folds:
        raise FileNotFoundError(f"no fold exports found under {export_root}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

    experiments = parse_list(args.experiments)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    per_seed_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    diagnostics: dict[str, Any] = {}
    leakage_report: dict[str, Any] = {}

    for fold in folds:
        export_path = export_root / f"k={fold}" / "mmp_latent_export.pt"
        export = load_export(export_path)
        train_obj = export["train"]
        test_obj = export["test"]
        required = ["sample_ids", "event_times", "censorships", "wsi_embedding", "rna_embedding"]
        for key in required:
            if key not in train_obj or key not in test_obj:
                raise KeyError(f"missing required export key {key} in fold {fold}")

        leakage_report[f"k={fold}"] = {
            "export_path": str(export_path),
            "sample_overlap": sample_overlap_report(train_obj["sample_ids"], test_obj["sample_ids"]),
        }

        wsi_train = train_obj["wsi_embedding"].astype(np.float32)
        rna_train = train_obj["rna_embedding"].astype(np.float32)
        wsi_test = test_obj["wsi_embedding"].astype(np.float32)
        rna_test = test_obj["rna_embedding"].astype(np.float32)
        train_times = train_obj["event_times"].astype(np.float32)
        train_censorships = train_obj["censorships"].astype(np.float32)
        test_times = test_obj["event_times"].astype(np.float32)
        test_censorships = test_obj["censorships"].astype(np.float32)

        wsi_dim = int(wsi_train.shape[1])
        rna_dim = int(rna_train.shape[1])
        diagnostics.setdefault(f"k={fold}", {})

        for experiment in experiments:
            if experiment == "mmp__original":
                fold_rows.append(
                    summarize_original_mmp_reference(
                        fold=fold,
                        train_obj=train_obj,
                        test_obj=test_obj,
                    )
                )
                continue

            model_factory = make_model_factory(
                experiment=experiment,
                wsi_dim=wsi_dim,
                rna_dim=rna_dim,
                hidden_dim=int(args.hidden_dim),
                dropout=float(args.dropout),
                num_points=int(args.num_points),
                tiny_num_points=int(args.tiny_num_points),
                tiny_compact_dim=int(args.tiny_compact_dim),
                tiny_sidecar_hidden_dim=int(args.tiny_sidecar_hidden_dim),
                tiny_residual_scale_init=float(args.tiny_residual_scale_init),
            )
            run_results: list[RunResult] = []
            last_artifacts: dict[str, object] | None = None

            for seed in seeds:
                result, artifacts = run_one_seed_pair(
                    fold=fold,
                    experiment=experiment,
                    seed=seed,
                    wsi_train=wsi_train,
                    rna_train=rna_train,
                    train_times=train_times,
                    train_censorships=train_censorships,
                    wsi_test=wsi_test,
                    rna_test=rna_test,
                    test_times=test_times,
                    test_censorships=test_censorships,
                    model_factory=model_factory,
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    val_frac=float(args.val_frac),
                    split_mode=str(args.split_mode),
                    num_time_bins=int(args.time_bins),
                    selection_metric=str(args.selection_metric),
                    min_epochs=int(args.min_epochs),
                    early_stop_patience=int(args.early_stop_patience),
                    out_dir=out_dir,
                )
                run_results.append(result)
                last_artifacts = artifacts
                per_seed_rows.append(
                    {
                        "fold": fold,
                        "experiment": result.experiment,
                        "seed": result.seed,
                        "best_epoch": result.best_epoch,
                        "best_val_c_index": result.best_val_c_index,
                        "train_c_index": result.train_c_index,
                        "val_c_index": result.val_c_index,
                        "test_c_index": result.test_c_index,
                    }
                )

            test_scores = np.array([r.test_c_index for r in run_results], dtype=np.float64)
            val_scores = np.array([r.val_c_index for r in run_results], dtype=np.float64)
            train_scores = np.array([r.train_c_index for r in run_results], dtype=np.float64)
            fold_row = {
                "fold": fold,
                "experiment": experiment,
                "num_seeds": len(run_results),
                "mean_val_c_index": float(val_scores.mean()),
                "std_val_c_index": float(val_scores.std(ddof=1)) if len(val_scores) > 1 else 0.0,
                "mean_test_c_index": float(test_scores.mean()),
                "std_test_c_index": float(test_scores.std(ddof=1)) if len(test_scores) > 1 else 0.0,
                "mean_train_c_index": float(train_scores.mean()),
            }
            fold_rows.append(fold_row)

            if experiment in {"fusion__geometry", "fusion__baseline_tiny_geometry"} and last_artifacts is not None:
                aux = last_artifacts["aux"]
                if "geometry_enhanced" in aux:
                    diagnostics[f"k={fold}"][experiment] = {
                        "top_geometry_correlations": top_geometry_correlations(
                            aux["geometry_enhanced"],
                            last_artifacts["test_pred"],
                        )
                    }

            if args.run_permutation_check and experiment in {"fusion__geometry", "fusion__gated"}:
                perm_seed = int(seeds[0])
                perm_times, perm_censorships = permute_survival_labels(train_times, train_censorships, perm_seed)
                perm_factory = make_model_factory(
                    experiment=experiment,
                    wsi_dim=wsi_dim,
                    rna_dim=rna_dim,
                    hidden_dim=int(args.hidden_dim),
                    dropout=float(args.dropout),
                    num_points=int(args.num_points),
                    tiny_num_points=int(args.tiny_num_points),
                    tiny_compact_dim=int(args.tiny_compact_dim),
                    tiny_sidecar_hidden_dim=int(args.tiny_sidecar_hidden_dim),
                    tiny_residual_scale_init=float(args.tiny_residual_scale_init),
                )
                perm_result, _ = run_one_seed_pair(
                    fold=fold,
                    experiment=f"{experiment}__permuted",
                    seed=perm_seed,
                    wsi_train=wsi_train,
                    rna_train=rna_train,
                    train_times=perm_times,
                    train_censorships=perm_censorships,
                    wsi_test=wsi_test,
                    rna_test=rna_test,
                    test_times=test_times,
                    test_censorships=test_censorships,
                    model_factory=perm_factory,
                    epochs=int(args.perm_epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    val_frac=float(args.val_frac),
                    split_mode=str(args.split_mode),
                    num_time_bins=int(args.time_bins),
                    selection_metric=str(args.selection_metric),
                    min_epochs=int(args.min_epochs),
                    early_stop_patience=int(args.early_stop_patience),
                    out_dir=out_dir / "permutation",
                )
                permutation_rows.append(
                    {
                        "fold": fold,
                        "experiment": experiment,
                        "perm_seed": perm_seed,
                        "perm_epochs": int(args.perm_epochs),
                        "permuted_train_c_index": perm_result.train_c_index,
                        "permuted_val_c_index": perm_result.val_c_index,
                        "permuted_test_c_index": perm_result.test_c_index,
                    }
                )

    for experiment in experiments:
        matching = [row for row in fold_rows if row["experiment"] == experiment]
        if not matching:
            continue
        fold_test_mean, fold_test_std = finite_mean_std([float(row["mean_test_c_index"]) for row in matching])
        fold_val_mean, fold_val_std = finite_mean_std([float(row["mean_val_c_index"]) for row in matching])
        fold_train_mean, _ = finite_mean_std([float(row["mean_train_c_index"]) for row in matching])
        overall_rows.append(
            {
                "experiment": experiment,
                "num_folds": len(matching),
                "mean_fold_val_c_index": fold_val_mean,
                "std_fold_val_c_index": fold_val_std,
                "mean_fold_test_c_index": fold_test_mean,
                "std_fold_test_c_index": fold_test_std,
                "mean_fold_train_c_index": fold_train_mean,
            }
        )

    write_csv(
        out_dir / "per_seed_results.csv",
        per_seed_rows,
        ["fold", "experiment", "seed", "best_epoch", "best_val_c_index", "train_c_index", "val_c_index", "test_c_index"],
    )
    write_csv(
        out_dir / "per_fold_results.csv",
        fold_rows,
        ["fold", "experiment", "num_seeds", "mean_val_c_index", "std_val_c_index", "mean_test_c_index", "std_test_c_index", "mean_train_c_index"],
    )
    write_csv(
        out_dir / "overall_results.csv",
        overall_rows,
        ["experiment", "num_folds", "mean_fold_val_c_index", "std_fold_val_c_index", "mean_fold_test_c_index", "std_fold_test_c_index", "mean_fold_train_c_index"],
    )
    if permutation_rows:
        write_csv(
            out_dir / "permutation_results.csv",
            permutation_rows,
            ["fold", "experiment", "perm_seed", "perm_epochs", "permuted_train_c_index", "permuted_val_c_index", "permuted_test_c_index"],
        )
    if diagnostics:
        save_json(out_dir / "diagnostics" / "geometry_diagnostics.json", diagnostics)
    save_json(out_dir / "leakage_report.json", leakage_report)

    baseline_reference = {
        "protocol": "matched 5-fold latent benchmark",
        "export_root": str(export_root),
        "folds": folds,
        "experiments": experiments,
        "overall_results": overall_rows,
        "recommended_small_project_baseline": next(
            (row for row in overall_rows if row["experiment"] == "fusion__baseline"),
            None,
        ),
    }
    save_json(out_dir / "baseline_reference.json", baseline_reference)

    summary = {
        "out_dir": str(out_dir),
        "export_root": str(export_root),
        "folds": folds,
        "experiments": experiments,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_frac": float(args.val_frac),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "num_points": int(args.num_points),
        "tiny_num_points": int(args.tiny_num_points),
        "tiny_compact_dim": int(args.tiny_compact_dim),
        "tiny_sidecar_hidden_dim": int(args.tiny_sidecar_hidden_dim),
        "tiny_residual_scale_init": float(args.tiny_residual_scale_init),
        "seeds": seeds,
        "split_mode": str(args.split_mode),
        "time_bins": int(args.time_bins),
        "selection_metric": str(args.selection_metric),
        "min_epochs": int(args.min_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "overall_results": overall_rows,
        "permutation_results": permutation_rows,
    }
    save_json(out_dir / "summary.json", summary)

    print(f"export_root={export_root}")
    print(f"results_dir={out_dir}")
    for row in overall_rows:
        if str(row["experiment"]) == "mmp__original":
            print(
                f"{row['experiment']}: "
                f"fold_test={row['mean_fold_test_c_index']:.4f} +- {row['std_fold_test_c_index']:.4f}, "
                "fold_val=reference_only"
            )
        else:
            print(
                f"{row['experiment']}: "
                f"fold_test={row['mean_fold_test_c_index']:.4f} +- {row['std_fold_test_c_index']:.4f}, "
                f"fold_val={row['mean_fold_val_c_index']:.4f} +- {row['std_fold_val_c_index']:.4f}"
            )


if __name__ == "__main__":
    main()
