from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def torch_load_compat(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_export(path: Path) -> dict[str, dict[str, np.ndarray]]:
    obj = torch_load_compat(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected top-level dict, got {type(obj).__name__}")
    export: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "test"):
        split_obj = obj.get(split_name)
        if not isinstance(split_obj, dict):
            raise TypeError(f"Missing split dict: {split_name}")
        export[split_name] = {key: to_numpy(value) for key, value in split_obj.items()}
    return export


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_val_indices(n: int, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    val_size = max(1, int(round(n * val_frac)))
    val_idx = np.sort(order[:val_size])
    train_idx = np.sort(order[val_size:])
    return train_idx, val_idx


def standardize(
    train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def neg_partial_log_likelihood(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    risk = risk[order].reshape(-1)
    events = events[order].reshape(-1)
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    diff = risk - log_cumsum
    denom = events.sum().clamp_min(1.0)
    return -(diff * events).sum() / denom


def concordance_index(risk: np.ndarray, times: np.ndarray, censorships: np.ndarray) -> float:
    events = 1.0 - censorships.astype(np.float32)
    concordant = 0.0
    comparable = 0.0
    n = len(risk)
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


def sample_overlap_report(train_ids: np.ndarray, test_ids: np.ndarray) -> dict[str, object]:
    train_set = {str(x) for x in train_ids.tolist()}
    test_set = {str(x) for x in test_ids.tolist()}
    overlap = sorted(train_set & test_set)
    return {
        "num_train": len(train_set),
        "num_test": len(test_set),
        "num_overlap": len(overlap),
        "overlap_sample_ids_head": overlap[:20],
    }


def permute_survival_labels(times: np.ndarray, censorships: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(times.shape[0])
    rng.shuffle(order)
    return times[order].copy(), censorships[order].copy()


class FusionConcatBaselineRisk(nn.Module):
    def __init__(self, wsi_dim: int, rna_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(wsi_dim + rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, wsi: torch.Tensor, rna: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = torch.cat([wsi, rna], dim=1)
        hidden = self.backbone(fused)
        return self.head(hidden).squeeze(-1), {"fused": fused}


class GatedFusionRisk(nn.Module):
    def __init__(self, wsi_dim: int, rna_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(wsi_dim, rna_dim), nn.Sigmoid())
        self.backbone = nn.Sequential(
            nn.Linear(wsi_dim + rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, wsi: torch.Tensor, rna: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate = self.gate_net(wsi)
        rna_mod = rna * gate
        fused = torch.cat([wsi, rna_mod], dim=1)
        hidden = self.backbone(fused)
        risk = self.head(hidden).squeeze(-1)
        return risk, {"fused": fused, "gate": gate}


@dataclass
class RunResult:
    experiment: str
    seed: int
    best_epoch: int
    best_val_c_index: float
    train_c_index: float
    val_c_index: float
    test_c_index: float


def run_one_seed_pair(
    *,
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
    out_dir: Path,
) -> tuple[RunResult, dict[str, object]]:
    set_seed(seed)
    fit_idx, val_idx = split_train_val_indices(wsi_train.shape[0], seed, val_frac)
    wsi_fit, wsi_val, wsi_test_std, wsi_mean, wsi_std = standardize(wsi_train[fit_idx], wsi_train[val_idx], wsi_test)
    rna_fit, rna_val, rna_test_std, rna_mean, rna_std = standardize(rna_train[fit_idx], rna_train[val_idx], rna_test)

    fit_times = train_times[fit_idx].astype(np.float32)
    fit_censorships = train_censorships[fit_idx].astype(np.float32)
    val_times = train_times[val_idx].astype(np.float32)
    val_censorships = train_censorships[val_idx].astype(np.float32)
    fit_events = 1.0 - fit_censorships

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

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | int] | None = None
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
            val_pred = model(wsi_val_t, rna_val_t)[0].cpu().numpy()
            test_pred = model(wsi_test_t, rna_test_t)[0].cpu().numpy()

        metrics = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "train_c_index": concordance_index(fit_pred, fit_times, fit_censorships),
            "val_c_index": concordance_index(val_pred, val_times, val_censorships),
            "test_c_index": concordance_index(test_pred, test_times, test_censorships),
        }
        history.append(metrics)
        if best_metrics is None or float(metrics["val_c_index"]) > float(best_metrics["val_c_index"]):
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
        out_dir / "histories" / f"{experiment}__seed{seed}.csv",
        history,
        ["epoch", "loss", "train_c_index", "val_c_index", "test_c_index"],
    )

    return (
        RunResult(
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
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compare current main-project fusion baseline vs frozen gated merge on exported MMP latents.")
    p.add_argument("--dump-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--run-permutation-check", action="store_true")
    p.add_argument("--perm-epochs", type=int, default=80)
    args = p.parse_args()

    dump_path = Path(args.dump_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)

    export = load_export(dump_path)
    train_obj = export["train"]
    test_obj = export["test"]
    required = ["sample_ids", "event_times", "censorships", "wsi_embedding", "rna_embedding"]
    for key in required:
        if key not in train_obj or key not in test_obj:
            raise KeyError(f"missing required export key: {key}")

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
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    experiments = ["fusion__baseline", "fusion__gated"]
    per_seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []

    leakage_report = {
        "sample_overlap": sample_overlap_report(train_obj["sample_ids"], test_obj["sample_ids"]),
    }
    save_json(out_dir / "leakage_report.json", leakage_report)

    for experiment in experiments:
        run_results: list[RunResult] = []
        for seed in seeds:
            if experiment == "fusion__baseline":
                model_factory = lambda: FusionConcatBaselineRisk(
                    wsi_dim=wsi_dim, rna_dim=rna_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)
                )
            else:
                model_factory = lambda: GatedFusionRisk(
                    wsi_dim=wsi_dim, rna_dim=rna_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)
                )

            result, _ = run_one_seed_pair(
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
                out_dir=out_dir,
            )
            run_results.append(result)
            per_seed_rows.append(
                {
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
        aggregate_rows.append(
            {
                "experiment": experiment,
                "num_seeds": len(run_results),
                "mean_val_c_index": float(val_scores.mean()),
                "std_val_c_index": float(val_scores.std(ddof=1)) if len(val_scores) > 1 else 0.0,
                "mean_test_c_index": float(test_scores.mean()),
                "std_test_c_index": float(test_scores.std(ddof=1)) if len(test_scores) > 1 else 0.0,
                "mean_train_c_index": float(train_scores.mean()),
            }
        )

        if args.run_permutation_check and experiment == "fusion__gated":
            perm_seed = int(seeds[0])
            perm_times, perm_censorships = permute_survival_labels(train_times, train_censorships, perm_seed)
            perm_factory = lambda: GatedFusionRisk(
                wsi_dim=wsi_dim, rna_dim=rna_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)
            )
            perm_result, _ = run_one_seed_pair(
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
                out_dir=out_dir / "permutation",
            )
            permutation_rows.append(
                {
                    "experiment": experiment,
                    "perm_seed": perm_seed,
                    "perm_epochs": int(args.perm_epochs),
                    "permuted_train_c_index": perm_result.train_c_index,
                    "permuted_val_c_index": perm_result.val_c_index,
                    "permuted_test_c_index": perm_result.test_c_index,
                }
            )

    write_csv(
        out_dir / "per_seed_results.csv",
        per_seed_rows,
        ["experiment", "seed", "best_epoch", "best_val_c_index", "train_c_index", "val_c_index", "test_c_index"],
    )
    write_csv(
        out_dir / "aggregate_results.csv",
        aggregate_rows,
        ["experiment", "num_seeds", "mean_val_c_index", "std_val_c_index", "mean_test_c_index", "std_test_c_index", "mean_train_c_index"],
    )
    if permutation_rows:
        write_csv(
            out_dir / "permutation_results.csv",
            permutation_rows,
            ["experiment", "perm_seed", "perm_epochs", "permuted_train_c_index", "permuted_val_c_index", "permuted_test_c_index"],
        )

    summary = {
        "dump_path": str(dump_path),
        "out_dir": str(out_dir),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_frac": float(args.val_frac),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "seeds": seeds,
        "experiments": experiments,
        "aggregate_results": aggregate_rows,
        "leakage_report": leakage_report,
        "permutation_results": permutation_rows,
    }
    save_json(out_dir / "summary.json", summary)

    print(f"dump_path={dump_path}")
    print(f"hello world")
    print(f"results_dir={out_dir}")
    for row in aggregate_rows:
        print(
            f"{row['experiment']}: "
            f"test={row['mean_test_c_index']:.4f} +- {row['std_test_c_index']:.4f}, "
            f"val={row['mean_val_c_index']:.4f} +- {row['std_val_c_index']:.4f}"
        )
    if permutation_rows:
        for row in permutation_rows:
            print(
                f"{row['experiment']} permuted_test_c_index={row['permuted_test_c_index']:.4f}"
            )


if __name__ == "__main__":
    main()
