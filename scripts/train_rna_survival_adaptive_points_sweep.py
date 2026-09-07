from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _path in (str(_SRC), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train_rna_survival_baseline_sweep import (
    build_loader,
    infer_cohort_from_split_dir,
    load_gene_id_to_symbol_map,
    load_official_split,
    neg_partial_log_likelihood,
    parse_csv_list,
    resolve_default_rna_tsv,
    resolve_gene_annotation_gtf,
    resolve_gene_sets_csv,
    save_json,
    set_seed,
    split_train_val_case_ids,
    write_csv,
)
from train_rna_survival_baseline_sweep import _dense_block, OmicsMLPEncoder, concordance_index, gather_labels
from train_rna_survival_geo_sweep import geometry_regularizers
from r2wsp.data import resolve_data_paths
from r2wsp.models.direct_survival import CurveGeometryEncoder


ADAPTIVE_METHOD_CHOICES = ("gumbel_count", "sparse_points", "learnable_lambda_sparse")


class MaskAwarePointSetEncoder(nn.Module):
    def __init__(self, point_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(3, int(point_feature_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.point_hidden_proj = nn.Sequential(
            nn.Linear(int(point_feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.point_attn_score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        return self.point_hidden_proj(self.point_encoder(points))

    def pool(
        self,
        point_tokens: torch.Tensor,
        *,
        pool_mode: str,
        point_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = point_tokens.shape
        if point_weights is None:
            point_weights = torch.ones((batch_size, num_points), dtype=point_tokens.dtype, device=point_tokens.device)
        point_weights = point_weights.clamp_min(1e-6)
        if str(pool_mode) == "attn":
            attn_logits = self.point_attn_score(point_tokens).squeeze(-1)
            attn_logits = attn_logits + point_weights.log()
            attn_probs = torch.softmax(attn_logits, dim=1)
            attn_probs = attn_probs * point_weights
            attn_probs = attn_probs / attn_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            case = torch.sum(point_tokens * attn_probs.unsqueeze(-1), dim=1)
            return case, attn_probs
        norm = point_weights / point_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        case = torch.sum(point_tokens * norm.unsqueeze(-1), dim=1)
        return case, norm


def build_prefix_mask(counts: list[int], max_points: int, device: torch.device) -> torch.Tensor:
    masks = []
    for count in counts:
        mask = torch.zeros(int(max_points), device=device, dtype=torch.float32)
        mask[: int(count)] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0)


class RNAAdaptivePointModel(nn.Module):
    def __init__(
        self,
        *,
        rna_dim: int,
        omic_sizes: list[int],
        hidden_dim: int,
        dropout: float,
        max_points: int,
        count_candidates: list[int],
        point_feature_dim: int,
        method: str,
        gumbel_tau: float,
        sparse_gate_temperature: float,
        plain_omics_pool: str,
        point_pool_mode: str,
        lambda_floor: float,
    ) -> None:
        super().__init__()
        if method not in ADAPTIVE_METHOD_CHOICES:
            raise ValueError(f"unsupported method: {method}")
        self.method = str(method)
        self.max_points = int(max_points)
        self.count_candidates = [int(x) for x in count_candidates]
        self.gumbel_tau = float(gumbel_tau)
        self.sparse_gate_temperature = float(sparse_gate_temperature)
        self.plain_omics_pool = str(plain_omics_pool)
        self.point_pool_mode = str(point_pool_mode)
        self.lambda_floor = float(lambda_floor)

        self.dense_encoder = _dense_block(int(rna_dim), int(hidden_dim), float(dropout))
        self.omics_encoder = OmicsMLPEncoder(omic_sizes=list(omic_sizes), hidden_dim=int(hidden_dim), dropout=float(dropout))
        self.dense_curve_encoder = CurveGeometryEncoder(int(hidden_dim), int(max_points))
        self.point_set_encoder = MaskAwarePointSetEncoder(int(point_feature_dim), int(hidden_dim), float(dropout))

        self.count_selector = (
            nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), len(self.count_candidates)),
            )
            if self.method == "gumbel_count"
            else None
        )
        self.point_gate_head = (
            nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(max_points)),
            )
            if self.method in {"sparse_points", "learnable_lambda_sparse"}
            else None
        )
        self.sparse_lambda_param = (
            nn.Parameter(torch.tensor(0.0))
            if self.method == "learnable_lambda_sparse"
            else None
        )
        self.backbone = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(int(hidden_dim), 1)

    def _gumbel_count_case(self, dense_case: torch.Tensor, points: torch.Tensor, point_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.count_selector is None:
            raise RuntimeError("gumbel_count requires count_selector")
        count_logits = self.count_selector(dense_case)
        if self.training:
            count_probs = F.gumbel_softmax(count_logits, tau=float(self.gumbel_tau), hard=True, dim=1)
        else:
            hard_idx = torch.argmax(count_logits, dim=1)
            count_probs = F.one_hot(hard_idx, num_classes=len(self.count_candidates)).to(dtype=dense_case.dtype)
        prefix_masks = build_prefix_mask(self.count_candidates, self.max_points, dense_case.device)
        candidate_features: list[torch.Tensor] = []
        for idx, _count in enumerate(self.count_candidates):
            mask = prefix_masks[idx].unsqueeze(0).expand(points.shape[0], -1)
            feat, _ = self.point_set_encoder.pool(point_tokens, pool_mode=self.point_pool_mode, point_weights=mask)
            candidate_features.append(feat)
        stacked = torch.stack(candidate_features, dim=1)
        geo_case = torch.sum(stacked * count_probs.unsqueeze(-1), dim=1)
        counts_tensor = torch.tensor(self.count_candidates, device=dense_case.device, dtype=dense_case.dtype)
        effective_points = torch.sum(count_probs * counts_tensor.unsqueeze(0), dim=1)
        point_weights = torch.matmul(count_probs, prefix_masks).clamp_max(1.0)
        aux = {
            "count_logits": count_logits,
            "count_probs": count_probs,
            "effective_points": effective_points,
            "geo_point_weights": point_weights,
            "adaptive_penalty": dense_case.new_tensor(0.0),
        }
        return geo_case, aux

    def _sparse_case(self, dense_case: torch.Tensor, _points: torch.Tensor, point_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.point_gate_head is None:
            raise RuntimeError("sparse method requires point_gate_head")
        gate_logits = self.point_gate_head(dense_case)
        gates = torch.sigmoid(gate_logits / max(float(self.sparse_gate_temperature), 1e-6))
        geo_case, point_weights = self.point_set_encoder.pool(point_tokens, pool_mode=self.point_pool_mode, point_weights=gates)
        effective_points = gates.sum(dim=1)
        sparse_penalty = gates.mean()
        aux = {
            "gate_logits": gate_logits,
            "point_gates": gates,
            "effective_points": effective_points,
            "geo_point_weights": point_weights,
        }
        if self.method == "learnable_lambda_sparse":
            if self.sparse_lambda_param is None:
                raise RuntimeError("learnable_lambda_sparse requires sparse_lambda_param")
            sparse_lambda = F.softplus(self.sparse_lambda_param) + float(self.lambda_floor)
            adaptive_penalty = sparse_lambda * sparse_penalty + (1e-3 / sparse_lambda)
            aux["sparse_lambda"] = sparse_lambda.expand_as(effective_points)
            aux["adaptive_penalty"] = adaptive_penalty
        else:
            aux["adaptive_penalty"] = sparse_penalty
        return geo_case, aux

    def forward(self, *, rna_vec: torch.Tensor, rna_omics: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        dense_case = self.dense_encoder(rna_vec.float())
        omics_case, omics_tokens, omics_weights = self.omics_encoder(rna_omics, pool_mode=self.plain_omics_pool)
        aux["rna_dense_case"] = dense_case
        aux["rna_omics_case"] = omics_case
        aux["rna_omics_tokens"] = omics_tokens
        aux["rna_omics_weights"] = omics_weights

        geometry = self.dense_curve_encoder(dense_case)
        aux["geo_points"] = geometry["points"]
        aux["geo_curvature"] = geometry["curvature"]
        aux["geo_torsion"] = geometry["torsion"]
        aux["geo_geometry_per_point"] = geometry["geometry_per_point"]
        aux["geo_curvature_per_point"] = geometry["curvature_per_point"]
        aux["geo_torsion_per_point"] = geometry["torsion_per_point"]
        point_tokens = self.point_set_encoder.encode(geometry["points"])

        if self.method == "gumbel_count":
            geo_case, adaptive_aux = self._gumbel_count_case(dense_case, geometry["points"], point_tokens)
        else:
            geo_case, adaptive_aux = self._sparse_case(dense_case, geometry["points"], point_tokens)
        aux.update(adaptive_aux)

        regs = geometry_regularizers(geometry["points"])
        aux.update(regs)
        fused_case = omics_case + geo_case
        aux["geo_case"] = geo_case
        aux["rna_fused"] = fused_case
        hidden = self.backbone(fused_case)
        risk = self.head(hidden).squeeze(-1)
        aux["rna_hidden"] = hidden
        return risk, aux


def evaluate(
    model: RNAAdaptivePointModel,
    loader,
    case_table: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    risks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    censorships: list[np.ndarray] = []
    losses: list[float] = []
    effective_points_means: list[float] = []
    sparse_lambda_means: list[float] = []
    count_probs_means: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_times_t, batch_events_t = gather_labels(batch.case_id, case_table, device)
            risk, aux = model(
                rna_vec=batch.rna_vec.to(device),
                rna_omics=[x.to(device) for x in batch.rna_omics],
            )
            batch_loss = neg_partial_log_likelihood(risk, batch_times_t, batch_events_t)
            losses.append(float(batch_loss.detach().cpu()))
            risks.append(risk.detach().cpu().numpy())
            times.append(np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32))
            censorships.append(np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32))
            effective_points_means.append(float(aux["effective_points"].detach().mean().cpu()))
            if "sparse_lambda" in aux:
                sparse_lambda_means.append(float(aux["sparse_lambda"].detach().mean().cpu()))
            if "count_probs" in aux:
                count_probs_means.append(aux["count_probs"].detach().mean(dim=0).cpu().numpy())
    risk_all = np.concatenate(risks, axis=0)
    time_all = np.concatenate(times, axis=0)
    censorship_all = np.concatenate(censorships, axis=0)
    aux_summary = {
        "effective_points_mean": float(np.mean(effective_points_means)) if effective_points_means else float("nan"),
        "sparse_lambda_mean": float(np.mean(sparse_lambda_means)) if sparse_lambda_means else float("nan"),
    }
    if count_probs_means:
        probs = np.stack(count_probs_means, axis=0).mean(axis=0)
        for idx, value in enumerate(probs.tolist()):
            aux_summary[f"count_prob_{idx}"] = float(value)
    return concordance_index(risk_all, time_all, censorship_all), float(np.mean(losses)), aux_summary


def train_one_run(
    *,
    split_dir: Path,
    target_col: str,
    rna_tsv_path: Path,
    gene_sets_csv: Path,
    gene_id_to_symbol: dict[str, str] | None,
    out_dir: Path,
    seed: int,
    method: str,
    count_candidates: list[int],
    max_points: int,
    hidden_dim: int,
    point_feature_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    val_split_mode: str,
    val_time_bins: int,
    selection_metric: str,
    val_ema_decay: float,
    selection_min_epochs: int,
    early_stop_patience: int,
    device_name: str,
    num_workers: int,
    pin_memory: bool,
    plain_omics_pool: str,
    point_pool_mode: str,
    smooth_l1_weight: float,
    smooth_l2_weight: float,
    curvature_reg_weight: float,
    torsion_reg_weight: float,
    sparse_weight: float,
    gumbel_tau: float,
    sparse_gate_temperature: float,
    lambda_floor: float,
) -> dict[str, object]:
    set_seed(seed)
    case_table, official_train_case_ids, test_case_ids = load_official_split(split_dir, target_col)
    train_case_ids, val_case_ids = split_train_val_case_ids(
        sorted(official_train_case_ids),
        int(seed),
        float(val_frac),
        case_table,
        mode=str(val_split_mode),
        n_time_bins=int(val_time_bins),
    )
    train_ds, train_loader = build_loader(
        sorted(train_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch="dense_omics",
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    _val_ds, val_loader = build_loader(
        sorted(val_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch="dense_omics",
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    _test_ds, test_loader = build_loader(
        sorted(test_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch="dense_omics",
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    model = RNAAdaptivePointModel(
        rna_dim=int(train_ds.rna_dim),
        omic_sizes=train_ds.omic_sizes or [],
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        max_points=int(max_points),
        count_candidates=list(count_candidates),
        point_feature_dim=int(point_feature_dim),
        method=str(method),
        gumbel_tau=float(gumbel_tau),
        sparse_gate_temperature=float(sparse_gate_temperature),
        plain_omics_pool=str(plain_omics_pool),
        point_pool_mode=str(point_pool_mode),
        lambda_floor=float(lambda_floor),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_val = float("-inf")
    best_test = float("nan")
    best_train = float("nan")
    best_val_loss = float("inf")
    best_select_score = float("-inf") if str(selection_metric) != "val_loss" else float("inf")
    val_c_index_ema: float | None = None
    stale_epochs = 0
    best_aux_summary: dict[str, float] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)
    print(
        f"[run] method={method} max_points={max_points} candidates={count_candidates} "
        f"seed={seed} out_dir={out_dir}"
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_risks: list[np.ndarray] = []
        train_times_np: list[np.ndarray] = []
        train_cens_np: list[np.ndarray] = []
        train_effective_points: list[float] = []
        train_sparse_lambda: list[float] = []
        for batch in train_loader:
            times, events = gather_labels(batch.case_id, case_table, device)
            risk, aux_train = model(
                rna_vec=batch.rna_vec.to(device),
                rna_omics=[x.to(device) for x in batch.rna_omics],
            )
            loss = neg_partial_log_likelihood(risk, times, events)
            loss = loss + float(smooth_l1_weight) * aux_train["smooth_l1"]
            loss = loss + float(smooth_l2_weight) * aux_train["smooth_l2"]
            loss = loss + float(curvature_reg_weight) * aux_train["curvature_l2"]
            loss = loss + float(torsion_reg_weight) * aux_train["torsion_l2"]
            adaptive_penalty = aux_train["adaptive_penalty"]
            if str(method) == "sparse_points":
                loss = loss + float(sparse_weight) * adaptive_penalty
            elif str(method) == "learnable_lambda_sparse":
                loss = loss + adaptive_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_risks.append(risk.detach().cpu().numpy())
            train_times_np.append(np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32))
            train_cens_np.append(np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32))
            train_effective_points.append(float(aux_train["effective_points"].detach().mean().cpu()))
            if "sparse_lambda" in aux_train:
                train_sparse_lambda.append(float(aux_train["sparse_lambda"].detach().mean().cpu()))

        train_c_index = concordance_index(
            np.concatenate(train_risks, axis=0),
            np.concatenate(train_times_np, axis=0),
            np.concatenate(train_cens_np, axis=0),
        )
        val_c_index, val_loss, val_aux = evaluate(model, val_loader, case_table, device)
        test_c_index, _, test_aux = evaluate(model, test_loader, case_table, device)

        if val_c_index_ema is None:
            val_c_index_ema = float(val_c_index)
        else:
            val_c_index_ema = float(val_ema_decay) * float(val_c_index_ema) + (1.0 - float(val_ema_decay)) * float(val_c_index)

        if str(selection_metric) == "val_loss":
            select_score = float(val_loss)
            improved = epoch >= int(selection_min_epochs) and (
                select_score < best_select_score if np.isfinite(best_select_score) else True
            )
        elif str(selection_metric) == "val_c_index_ema":
            select_score = float(val_c_index_ema)
            improved = epoch >= int(selection_min_epochs) and select_score >= float(best_select_score)
        else:
            select_score = float(val_c_index)
            improved = epoch >= int(selection_min_epochs) and select_score >= float(best_select_score)

        if improved:
            best_val = float(val_c_index)
            best_val_loss = float(val_loss)
            best_test = float(test_c_index)
            best_train = float(train_c_index)
            best_epoch = epoch
            best_select_score = float(select_score)
            best_aux_summary = dict(test_aux)
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "method": str(method),
                    "count_candidates": list(count_candidates),
                    "max_points": int(max_points),
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_val_c_index": float(best_val),
                    "best_test_c_index": float(best_test),
                },
                out_dir / "best.pt",
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "loss": float(np.mean(train_losses)),
            "train_c_index": float(train_c_index),
            "val_c_index": float(val_c_index),
            "val_loss": float(val_loss),
            "val_c_index_ema": float(val_c_index_ema),
            "selection_metric": str(selection_metric),
            "selection_score": float(select_score),
            "test_c_index": float(test_c_index),
            "train_effective_points": float(np.mean(train_effective_points)) if train_effective_points else float("nan"),
            "val_effective_points": float(val_aux.get("effective_points_mean", float("nan"))),
            "test_effective_points": float(test_aux.get("effective_points_mean", float("nan"))),
            "train_sparse_lambda": float(np.mean(train_sparse_lambda)) if train_sparse_lambda else float("nan"),
            "test_sparse_lambda": float(test_aux.get("sparse_lambda_mean", float("nan"))),
        }
        history_rows.append(row)
        print(
            f"epoch={epoch} method={method} seed={seed} loss={row['loss']:.4f} "
            f"train_c_index={row['train_c_index']:.4f} val_c_index={row['val_c_index']:.4f} "
            f"test_c_index={row['test_c_index']:.4f} test_eff_points={row['test_effective_points']:.2f}"
        )
        if epoch >= int(selection_min_epochs) and stale_epochs >= int(early_stop_patience):
            print(f"early_stop method={method} seed={seed} epoch={epoch} best_epoch={best_epoch}")
            break

    write_csv(
        out_dir / "histories" / f"seed{int(seed)}.csv",
        history_rows,
        [
            "epoch",
            "loss",
            "train_c_index",
            "val_c_index",
            "val_loss",
            "val_c_index_ema",
            "selection_metric",
            "selection_score",
            "test_c_index",
            "train_effective_points",
            "val_effective_points",
            "test_effective_points",
            "train_sparse_lambda",
            "test_sparse_lambda",
        ],
    )

    summary = {
        "mode": "rna_adaptive_points_survival",
        "seed": int(seed),
        "method": str(method),
        "count_candidates": list(count_candidates),
        "max_points": int(max_points),
        "best_epoch": int(best_epoch),
        "best_val_c_index": float(best_val),
        "best_test_c_index": float(best_test),
        "best_train_c_index": float(best_train),
        "best_val_loss": float(best_val_loss),
        "best_select_score": float(best_select_score),
        "out_dir": str(out_dir),
        "rna_tsv_path": str(rna_tsv_path),
        "rna_gene_sets_csv": str(gene_sets_csv),
    }
    summary.update(best_aux_summary)
    save_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive point-count RNA geo sweep.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--rna_tsv", default=None)
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--methods", default="gumbel_count,sparse_points,learnable_lambda_sparse")
    p.add_argument("--count_candidates", default="24,27,28,32,36,39,40")
    p.add_argument("--max_points", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--val_split_mode", default="survival_stratified", choices=["random", "survival_stratified"])
    p.add_argument("--val_time_bins", type=int, default=4)
    p.add_argument("--selection_metric", default="val_c_index_ema", choices=["val_c_index", "val_c_index_ema", "val_loss"])
    p.add_argument("--val_ema_decay", type=float, default=0.6)
    p.add_argument("--selection_min_epochs", type=int, default=8)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--point_feature_dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--smooth_l1_weight", type=float, default=0.0)
    p.add_argument("--smooth_l2_weight", type=float, default=0.0)
    p.add_argument("--curvature_reg_weight", type=float, default=0.0)
    p.add_argument("--torsion_reg_weight", type=float, default=0.0)
    p.add_argument("--sparse_weight", type=float, default=5e-3)
    p.add_argument("--gumbel_tau", type=float, default=0.8)
    p.add_argument("--sparse_gate_temperature", type=float, default=0.7)
    p.add_argument("--lambda_floor", type=float, default=1e-4)
    p.add_argument("--plain_omics_pool", default="attn", choices=["mean", "attn"])
    p.add_argument("--point_pool_mode", default="attn", choices=["mean", "attn"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--run_tag", default="adaptive_points_sweep")
    p.add_argument("--out_dir", default="outputs/rna_adaptive_points")
    args = p.parse_args()

    split_dir = Path(args.split_dir).resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"split_dir not found: {split_dir}")
    methods = parse_csv_list(args.methods, default=list(ADAPTIVE_METHOD_CHOICES))
    if any(x not in ADAPTIVE_METHOD_CHOICES for x in methods):
        raise ValueError(f"unsupported methods: {methods}")
    count_candidates = [int(x) for x in parse_csv_list(args.count_candidates, default=["24", "27", "28", "32", "36", "39", "40"])]
    seeds = [int(x) for x in parse_csv_list(args.seeds, default=[str(int(args.seed))])]

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()
    cohort = infer_cohort_from_split_dir(split_dir)
    rna_tsv_path = Path(args.rna_tsv).resolve() if args.rna_tsv is not None else resolve_default_rna_tsv(paths.raw_rna_root, cohort)
    gene_sets_csv = resolve_gene_sets_csv(paths.raw_rna_root, args.rna_gene_sets_csv)
    gtf_path = resolve_gene_annotation_gtf(args.gene_annotation_gtf)
    gene_id_to_symbol = load_gene_id_to_symbol_map(gtf_path) if gtf_path is not None else None

    run_root = (Path(args.out_dir).resolve() / split_dir.name / str(args.run_tag)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] split_dir={split_dir}")
    print(f"[sweep] rna_tsv_path={rna_tsv_path}")
    print(f"[sweep] gene_sets_csv={gene_sets_csv}")
    print(f"[sweep] gene_annotation_gtf={gtf_path}")
    print(f"[sweep] run_root={run_root}")

    leaderboard_rows: list[dict[str, object]] = []
    for seed in seeds:
        for method in methods:
            out_dir = run_root / f"method={method}" / f"seed={int(seed)}"
            summary = train_one_run(
                split_dir=split_dir,
                target_col=str(args.target_col),
                rna_tsv_path=rna_tsv_path,
                gene_sets_csv=gene_sets_csv,
                gene_id_to_symbol=gene_id_to_symbol,
                out_dir=out_dir,
                seed=int(seed),
                method=str(method),
                count_candidates=list(count_candidates),
                max_points=int(args.max_points),
                hidden_dim=int(args.hidden_dim),
                point_feature_dim=int(args.point_feature_dim),
                dropout=float(args.dropout),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                val_frac=float(args.val_frac),
                val_split_mode=str(args.val_split_mode),
                val_time_bins=int(args.val_time_bins),
                selection_metric=str(args.selection_metric),
                val_ema_decay=float(args.val_ema_decay),
                selection_min_epochs=int(args.selection_min_epochs),
                early_stop_patience=int(args.early_stop_patience),
                device_name=str(args.device),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                plain_omics_pool=str(args.plain_omics_pool),
                point_pool_mode=str(args.point_pool_mode),
                smooth_l1_weight=float(args.smooth_l1_weight),
                smooth_l2_weight=float(args.smooth_l2_weight),
                curvature_reg_weight=float(args.curvature_reg_weight),
                torsion_reg_weight=float(args.torsion_reg_weight),
                sparse_weight=float(args.sparse_weight),
                gumbel_tau=float(args.gumbel_tau),
                sparse_gate_temperature=float(args.sparse_gate_temperature),
                lambda_floor=float(args.lambda_floor),
            )
            leaderboard_rows.append(summary)
            fieldnames = [
                "seed",
                "method",
                "best_epoch",
                "best_val_c_index",
                "best_test_c_index",
                "best_train_c_index",
                "best_val_loss",
                "best_select_score",
                "effective_points_mean",
                "sparse_lambda_mean",
                "out_dir",
            ]
            leaderboard_export = [{key: row.get(key) for key in fieldnames} for row in sorted(leaderboard_rows, key=lambda x: (float(x["best_val_c_index"]), float(x["best_test_c_index"])), reverse=True)]
            write_csv(run_root / "leaderboard.csv", leaderboard_export, fieldnames)

    save_json(
        run_root / "summary.json",
        {
            "mode": "rna_adaptive_points_sweep",
            "split_dir": str(split_dir),
            "run_root": str(run_root),
            "runs": leaderboard_rows,
        },
    )
    print(f"[done] run_root={run_root}")


if __name__ == "__main__":
    main()
