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
from torch.nn import functional as F


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


def gaussian_smooth_1d(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] < 3:
        return values
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=values.device, dtype=values.dtype)
    kernel = kernel / kernel.sum()
    padded = F.pad(values.unsqueeze(1), (2, 2), mode="replicate")
    smoothed = F.conv1d(padded, kernel.view(1, 1, -1))
    return smoothed.squeeze(1)


def signal_entropy(values: torch.Tensor) -> torch.Tensor:
    probs = values.abs() + 1e-6
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(probs * probs.log()).sum(dim=1)


def sequence_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    min_len = min(x.shape[1], y.shape[1])
    if min_len <= 1:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    x = x[:, :min_len]
    y = y[:, :min_len]
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    denom = torch.sqrt((x.pow(2).sum(dim=1) * y.pow(2).sum(dim=1)).clamp_min(1e-8))
    return (x * y).sum(dim=1) / denom


def quantile_features(values: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    q = torch.tensor(quantiles, device=values.device, dtype=values.dtype)
    return torch.quantile(values, q, dim=1).transpose(0, 1)


def coordinate_feature_bank(points: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]
    radius = torch.sqrt(x.pow(2) + y.pow(2) + eps)
    rho = torch.sqrt(x.pow(2) + y.pow(2) + z.pow(2) + eps)
    elevation = torch.atan2(z, radius + eps)
    return torch.stack(
        [
            radius.mean(dim=1),
            radius.std(dim=1, unbiased=False),
            rho.mean(dim=1),
            elevation.abs().mean(dim=1),
        ],
        dim=1,
    )


def local_geometry(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    r_prime = (points[:, 2:, :] - points[:, :-2, :]) / 2.0
    r_double = points[:, 2:, :] - 2.0 * points[:, 1:-1, :] + points[:, :-2, :]
    cross = torch.cross(r_prime, r_double, dim=-1)
    norm_cross = torch.norm(cross, dim=-1)
    norm_rp = torch.norm(r_prime, dim=-1)
    curvature = torch.clamp(norm_cross / (norm_rp.pow(3) + eps), min=0.0)

    if points.shape[1] >= 5:
        r_triple = points[:, 4:, :] - 3.0 * points[:, 3:-1, :] + 3.0 * points[:, 2:-2, :] - points[:, 1:-3, :]
        cross_tau = cross[:, 1:-1, :]
        norm_cross_tau = torch.norm(cross_tau, dim=-1)
        torsion = -torch.sum(cross_tau * r_triple, dim=-1) / (norm_cross_tau.pow(2) + eps)
    else:
        torsion = torch.zeros(points.shape[0], 1, device=points.device, dtype=points.dtype)
    return curvature, torsion


def multiscale_geometry_means(points: torch.Tensor, scales: tuple[int, ...] = (2, 4)) -> tuple[torch.Tensor, torch.Tensor]:
    curv_feats: list[torch.Tensor] = []
    tors_feats: list[torch.Tensor] = []
    for scale in scales:
        sampled = points[:, ::scale, :]
        if sampled.shape[1] < 5:
            curv_feats.append(torch.zeros(points.shape[0], device=points.device, dtype=points.dtype))
            tors_feats.append(torch.zeros(points.shape[0], device=points.device, dtype=points.dtype))
            continue
        curv_s, tors_s = local_geometry(sampled)
        curv_s = gaussian_smooth_1d(curv_s)
        tors_s = gaussian_smooth_1d(tors_s.abs())
        curv_feats.append(curv_s.mean(dim=1))
        tors_feats.append(tors_s.mean(dim=1))
    return torch.stack(curv_feats, dim=1), torch.stack(tors_feats, dim=1)


def summarize_enhanced_geometry_stats(points: torch.Tensor, curvature: torch.Tensor, torsion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    curvature_s = gaussian_smooth_1d(curvature)
    torsion_abs = gaussian_smooth_1d(torsion.abs())
    curv_q = quantile_features(curvature_s, [0.25, 0.5, 0.75])
    tors_q = quantile_features(torsion_abs, [0.5, 0.75])
    curv_ms, tors_ms = multiscale_geometry_means(points, scales=(2, 4))
    coord_feats = coordinate_feature_bank(points)
    corr_feat = sequence_corr(curvature_s, torsion_abs).unsqueeze(1)

    curvature_enhanced = torch.cat(
        [
            torch.stack(
                [
                    curvature_s.mean(dim=1),
                    curvature_s.std(dim=1, unbiased=False),
                    curvature_s.max(dim=1).values,
                    curvature_s.min(dim=1).values,
                    signal_entropy(curvature_s),
                ],
                dim=1,
            ),
            curv_q,
            curv_ms,
            coord_feats[:, :2],
        ],
        dim=1,
    )
    geometry_enhanced = torch.cat(
        [
            curvature_enhanced,
            torch.stack(
                [
                    torsion_abs.mean(dim=1),
                    torsion_abs.std(dim=1, unbiased=False),
                    torsion_abs.max(dim=1).values,
                    signal_entropy(torsion_abs),
                ],
                dim=1,
            ),
            tors_q,
            tors_ms,
            coord_feats[:, 2:],
            corr_feat,
        ],
        dim=1,
    )
    return curvature_enhanced, geometry_enhanced


GEOMETRY_ENHANCED_NAMES = [
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "curvature_min",
    "curv_entropy",
    "curv_q25",
    "curv_q50",
    "curv_q75",
    "curv_ms2_mean",
    "curv_ms4_mean",
    "geom_radius_mean",
    "geom_radius_std",
    "torsion_abs_mean",
    "torsion_abs_std",
    "torsion_abs_max",
    "tors_entropy",
    "tors_q50",
    "tors_q75",
    "tors_ms2_mean",
    "tors_ms4_mean",
    "geom_rho_mean",
    "geom_elevation_abs_mean",
    "geom_corr",
]


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


class GeometrySidecarRisk(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_points: int, dropout: float) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.delta_proj = nn.Linear(input_dim, num_points * 3)
        self.anchor_proj = nn.Linear(input_dim, 3)
        self.head = nn.Linear(hidden_dim, 1)
        self.sidecar = nn.Sequential(nn.Linear(23, 16), nn.ReLU(), nn.Linear(16, 1))

    def encode_geometry(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        deltas = self.delta_proj(x).view(bsz, -1, 3)
        points = torch.cumsum(deltas, dim=1) + self.anchor_proj(x).unsqueeze(1)
        curvature, torsion = local_geometry(points)
        _, geometry_enhanced = summarize_enhanced_geometry_stats(points, curvature, torsion)
        return geometry_enhanced, points

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.base(x)
        geometry_enhanced, points = self.encode_geometry(x)
        risk = self.head(hidden).squeeze(-1) + self.sidecar(geometry_enhanced).squeeze(-1)
        return risk, {"geometry_enhanced": geometry_enhanced, "points": points}


class FusionGeometryRisk(nn.Module):
    def __init__(self, wsi_dim: int, rna_dim: int, hidden_dim: int, num_points: int, dropout: float) -> None:
        super().__init__()
        self.inner = GeometrySidecarRisk(wsi_dim + rna_dim, hidden_dim, num_points, dropout)

    def forward(self, wsi: torch.Tensor, rna: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = torch.cat([wsi, rna], dim=1)
        risk, aux = self.inner(fused)
        aux["fused"] = fused
        return risk, aux


class FusionBaselineTinyGeometryRisk(nn.Module):
    def __init__(
        self,
        wsi_dim: int,
        rna_dim: int,
        hidden_dim: int,
        num_points: int,
        dropout: float,
        compact_dim: int = 32,
        sidecar_hidden_dim: int = 8,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        input_dim = wsi_dim + rna_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.compact_proj = nn.Sequential(
            nn.Linear(input_dim, compact_dim),
            nn.ReLU(),
        )
        self.delta_proj = nn.Linear(compact_dim, num_points * 3)
        self.anchor_proj = nn.Linear(compact_dim, 3)
        self.sidecar = nn.Sequential(
            nn.Linear(23, sidecar_hidden_dim),
            nn.ReLU(),
            nn.Linear(sidecar_hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def encode_geometry(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        compact = self.compact_proj(fused)
        bsz = compact.shape[0]
        deltas = self.delta_proj(compact).view(bsz, -1, 3)
        points = torch.cumsum(deltas, dim=1) + self.anchor_proj(compact).unsqueeze(1)
        curvature, torsion = local_geometry(points)
        _, geometry_enhanced = summarize_enhanced_geometry_stats(points, curvature, torsion)
        return geometry_enhanced, points

    def forward(self, wsi: torch.Tensor, rna: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = torch.cat([wsi, rna], dim=1)
        hidden = self.backbone(fused)
        base_risk = self.head(hidden).squeeze(-1)
        geometry_enhanced, points = self.encode_geometry(fused)
        residual = self.sidecar(geometry_enhanced).squeeze(-1)
        risk = base_risk + self.residual_scale * residual
        return risk, {
            "fused": fused,
            "geometry_enhanced": geometry_enhanced,
            "points": points,
            "base_risk": base_risk,
            "geometry_residual": residual,
            "residual_scale": self.residual_scale.unsqueeze(0),
        }


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
        },
    )


def top_geometry_correlations(geometry: np.ndarray, pred: np.ndarray, topk: int = 8) -> list[dict[str, float]]:
    def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
        if denom < 1e-12:
            return 0.0
        return float((x * y).sum() / denom)

    pairs = [
        {"feature": name, "corr_pred_risk": pearson_corr(geometry[:, idx], pred)}
        for idx, name in enumerate(GEOMETRY_ENHANCED_NAMES)
    ]
    pairs.sort(key=lambda item: abs(item["corr_pred_risk"]), reverse=True)
    return pairs[:topk]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare current main-project fusion baseline vs explicit fusion__geometry on exported MMP latents.")
    p.add_argument("--dump-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num-points", type=int, default=24)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--run-permutation-check", action="store_true")
    p.add_argument("--perm-epochs", type=int, default=80)
    args = p.parse_args()

    dump_path = Path(args.dump_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

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
    experiments = ["fusion__baseline", "fusion__geometry"]

    leakage_report = {
        "sample_overlap": sample_overlap_report(train_obj["sample_ids"], test_obj["sample_ids"]),
    }
    save_json(out_dir / "leakage_report.json", leakage_report)

    per_seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    diagnostics: dict[str, Any] = {}

    for experiment in experiments:
        run_results: list[RunResult] = []
        last_artifacts: dict[str, object] | None = None
        for seed in seeds:
            if experiment == "fusion__baseline":
                model_factory = lambda: FusionConcatBaselineRisk(
                    wsi_dim=wsi_dim, rna_dim=rna_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)
                )
            else:
                model_factory = lambda: FusionGeometryRisk(
                    wsi_dim=wsi_dim,
                    rna_dim=rna_dim,
                    hidden_dim=int(args.hidden_dim),
                    num_points=int(args.num_points),
                    dropout=float(args.dropout),
                )

            result, artifacts = run_one_seed_pair(
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
            last_artifacts = artifacts
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

        if experiment == "fusion__geometry" and last_artifacts is not None:
            aux = last_artifacts["aux"]
            if "geometry_enhanced" in aux:
                diagnostics[experiment] = {
                    "top_geometry_correlations": top_geometry_correlations(aux["geometry_enhanced"], last_artifacts["test_pred"]),
                }

        if args.run_permutation_check and experiment == "fusion__geometry":
            perm_seed = int(seeds[0])
            perm_times, perm_censorships = permute_survival_labels(train_times, train_censorships, perm_seed)
            perm_factory = lambda: FusionGeometryRisk(
                wsi_dim=wsi_dim,
                rna_dim=rna_dim,
                hidden_dim=int(args.hidden_dim),
                num_points=int(args.num_points),
                dropout=float(args.dropout),
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
    if diagnostics:
        save_json(out_dir / "diagnostics" / "geometry_diagnostics.json", diagnostics)

    summary = {
        "dump_path": str(dump_path),
        "out_dir": str(out_dir),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_frac": float(args.val_frac),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "num_points": int(args.num_points),
        "seeds": seeds,
        "experiments": experiments,
        "aggregate_results": aggregate_rows,
        "leakage_report": leakage_report,
        "permutation_results": permutation_rows,
        "diagnostics": diagnostics,
    }
    save_json(out_dir / "summary.json", summary)

    print(f"dump_path={dump_path}")
    print(f"results_dir={out_dir}")
    for row in aggregate_rows:
        print(
            f"{row['experiment']}: "
            f"test={row['mean_test_c_index']:.4f} +- {row['std_test_c_index']:.4f}, "
            f"val={row['mean_val_c_index']:.4f} +- {row['std_val_c_index']:.4f}"
        )
    if permutation_rows:
        for row in permutation_rows:
            print(f"{row['experiment']} permuted_test_c_index={row['permuted_test_c_index']:.4f}")


if __name__ == "__main__":
    main()
