from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
from r2wsp.data import resolve_data_paths
from r2wsp.models.direct_survival import (
    CurveGeometryEncoder,
    TokenPointGeometryEncoder,
    geometry_per_point_tensor,
    local_geometry,
)
from train_rna_survival_baseline_sweep import _dense_block, OmicsMLPEncoder, concordance_index, gather_labels


COMBO_CHOICES = ("dense_geo_plus_omics", "omics_geo_plus_dense")
GEO_IMPL_CHOICES = ("points", "curve")
GEO_FUSION_CHOICES = ("concat", "add", "gated")
GEO_POOL_CHOICES = ("mean", "attn")


def geometry_regularizers(points: torch.Tensor) -> dict[str, torch.Tensor]:
    d1 = points[:, 1:, :] - points[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    curvature, torsion = local_geometry(points)
    return {
        "smooth_l1": (d1 ** 2).mean(),
        "smooth_l2": (d2 ** 2).mean(),
        "curvature_l2": curvature.pow(2).mean(),
        "torsion_l2": torsion.pow(2).mean(),
    }


class PointSetEncoder(nn.Module):
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

    def forward(self, points: torch.Tensor, pool_mode: str = "mean") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_tokens = self.point_hidden_proj(self.point_encoder(points))
        if str(pool_mode) == "attn":
            attn_logits = self.point_attn_score(point_tokens).squeeze(-1)
            point_weights = torch.softmax(attn_logits, dim=1)
            case = torch.sum(point_tokens * point_weights.unsqueeze(-1), dim=1)
        else:
            point_weights = torch.full(
                (point_tokens.shape[0], point_tokens.shape[1]),
                1.0 / float(point_tokens.shape[1]),
                dtype=point_tokens.dtype,
                device=point_tokens.device,
            )
            case = point_tokens.mean(dim=1)
        return case, point_tokens, point_weights


class RNAGeoSurvivalModel(nn.Module):
    def __init__(
        self,
        *,
        rna_dim: int,
        omic_sizes: list[int],
        hidden_dim: int,
        dropout: float,
        combo_mode: str,
        geo_impl: str,
        num_points: int,
        geo_fusion: str,
        geo_pool_mode: str,
        point_feature_dim: int,
        plain_omics_pool: str,
    ) -> None:
        super().__init__()
        if combo_mode not in COMBO_CHOICES:
            raise ValueError(f"unsupported combo_mode: {combo_mode}")
        if geo_impl not in GEO_IMPL_CHOICES:
            raise ValueError(f"unsupported geo_impl: {geo_impl}")
        if geo_fusion not in GEO_FUSION_CHOICES:
            raise ValueError(f"unsupported geo_fusion: {geo_fusion}")
        if geo_pool_mode not in GEO_POOL_CHOICES:
            raise ValueError(f"unsupported geo_pool_mode: {geo_pool_mode}")
        if plain_omics_pool not in {"mean", "attn"}:
            raise ValueError("plain_omics_pool must be mean|attn")

        self.combo_mode = str(combo_mode)
        self.geo_impl = str(geo_impl)
        self.num_points = int(num_points)
        self.geo_fusion = str(geo_fusion)
        self.geo_pool_mode = str(geo_pool_mode)
        self.plain_omics_pool = str(plain_omics_pool)

        self.dense_encoder = _dense_block(int(rna_dim), int(hidden_dim), float(dropout))
        self.omics_encoder = OmicsMLPEncoder(omic_sizes=list(omic_sizes), hidden_dim=int(hidden_dim), dropout=float(dropout))
        self.point_set_encoder = PointSetEncoder(int(point_feature_dim), int(hidden_dim), float(dropout))

        self.dense_curve_encoder = CurveGeometryEncoder(int(hidden_dim), int(num_points))
        self.omics_token_geo_encoder = TokenPointGeometryEncoder(int(hidden_dim), int(num_points))
        self.curve_enhanced_proj = nn.Sequential(
            nn.Linear(23, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        self.concat_proj = (
            _dense_block(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
            if self.geo_fusion == "concat"
            else None
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Sigmoid(),
            )
            if self.geo_fusion == "gated"
            else None
        )
        self.backbone = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(int(hidden_dim), 1)

    def _encode_geo_from_dense(self, dense_case: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        geometry = self.dense_curve_encoder(dense_case)
        aux = {f"geo_{k}": v for k, v in geometry.items()}
        if self.geo_impl == "points":
            geo_case, point_tokens, point_weights = self.point_set_encoder(geometry["points"], pool_mode=self.geo_pool_mode)
            aux["geo_point_tokens"] = point_tokens
            aux["geo_point_weights"] = point_weights
        else:
            geo_case = self.curve_enhanced_proj(geometry["geometry_enhanced"])
        return geo_case, aux

    def _encode_geo_from_omics(self, omics_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = torch.ones((omics_tokens.shape[0], omics_tokens.shape[1]), dtype=omics_tokens.dtype, device=omics_tokens.device)
        geometry = self.omics_token_geo_encoder(omics_tokens, mask, return_attn=True)
        aux = {f"geo_{k}": v for k, v in geometry.items()}
        if self.geo_impl == "points":
            geo_case, point_tokens, point_weights = self.point_set_encoder(geometry["points"], pool_mode=self.geo_pool_mode)
            aux["geo_point_tokens"] = point_tokens
            aux["geo_point_weights"] = point_weights
        else:
            geo_case = self.curve_enhanced_proj(geometry["geometry_enhanced"])
        return geo_case, aux

    def _fuse(self, plain_case: torch.Tensor, geo_case: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.geo_fusion == "concat":
            if self.concat_proj is None:
                raise RuntimeError("concat fusion requires concat_proj")
            fused = self.concat_proj(torch.cat([plain_case, geo_case], dim=1))
        elif self.geo_fusion == "add":
            fused = plain_case + geo_case
        else:
            if self.gate_mlp is None:
                raise RuntimeError("gated fusion requires gate_mlp")
            gate = self.gate_mlp(torch.cat([plain_case, geo_case], dim=1))
            fused = gate * plain_case + (1.0 - gate) * geo_case
            aux["geo_fusion_gate"] = gate
        return fused, aux

    def forward(self, *, rna_vec: torch.Tensor, rna_omics: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        dense_case = self.dense_encoder(rna_vec.float())
        omics_case, omics_tokens, omics_weights = self.omics_encoder(rna_omics, pool_mode=self.plain_omics_pool)
        aux["rna_dense_case"] = dense_case
        aux["rna_omics_case"] = omics_case
        aux["rna_omics_tokens"] = omics_tokens
        aux["rna_omics_weights"] = omics_weights

        if self.combo_mode == "dense_geo_plus_omics":
            geo_case, geo_aux = self._encode_geo_from_dense(dense_case)
            plain_case = omics_case
            aux["plain_case"] = omics_case
            aux["geo_source_case"] = dense_case
        else:
            geo_case, geo_aux = self._encode_geo_from_omics(omics_tokens)
            plain_case = dense_case
            aux["plain_case"] = dense_case
            aux["geo_source_case"] = omics_case

        aux.update(geo_aux)
        fused_case, fuse_aux = self._fuse(plain_case, geo_case)
        aux.update(fuse_aux)
        aux["geo_case"] = geo_case
        aux["rna_fused"] = fused_case

        if "geo_points" in aux:
            regs = geometry_regularizers(aux["geo_points"])
            aux.update(regs)
            curvature, torsion = local_geometry(aux["geo_points"])
            curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(aux["geo_points"])
            aux["geo_curvature"] = curvature
            aux["geo_torsion"] = torsion
            aux["geo_curvature_per_point"] = curvature_full
            aux["geo_torsion_per_point"] = torsion_full
            aux["geo_geometry_per_point"] = geometry_per_point
        else:
            aux["smooth_l1"] = fused_case.new_tensor(0.0)
            aux["smooth_l2"] = fused_case.new_tensor(0.0)
            aux["curvature_l2"] = fused_case.new_tensor(0.0)
            aux["torsion_l2"] = fused_case.new_tensor(0.0)

        hidden = self.backbone(fused_case)
        risk = self.head(hidden).squeeze(-1)
        aux["rna_hidden"] = hidden
        return risk, aux


def evaluate(
    model: RNAGeoSurvivalModel,
    loader,
    case_table: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    risks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    censorships: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_times_t, batch_events_t = gather_labels(batch.case_id, case_table, device)
            risk, _ = model(
                rna_vec=batch.rna_vec.to(device),
                rna_omics=[x.to(device) for x in batch.rna_omics],
            )
            batch_loss = neg_partial_log_likelihood(risk, batch_times_t, batch_events_t)
            losses.append(float(batch_loss.detach().cpu()))
            risks.append(risk.detach().cpu().numpy())
            times.append(np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32))
            censorships.append(np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32))
    risk_all = np.concatenate(risks, axis=0)
    time_all = np.concatenate(times, axis=0)
    censorship_all = np.concatenate(censorships, axis=0)
    return concordance_index(risk_all, time_all, censorship_all), float(np.mean(losses))


def train_one_run(
    *,
    split_dir: Path,
    target_col: str,
    rna_tsv_path: Path,
    gene_sets_csv: Path,
    gene_id_to_symbol: dict[str, str] | None,
    out_dir: Path,
    seed: int,
    combo_mode: str,
    geo_impl: str,
    num_points: int,
    geo_fusion: str,
    geo_pool_mode: str,
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
    smooth_l1_weight: float,
    smooth_l2_weight: float,
    curvature_reg_weight: float,
    torsion_reg_weight: float,
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
    val_ds, val_loader = build_loader(
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
    test_ds, test_loader = build_loader(
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

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = RNAGeoSurvivalModel(
        rna_dim=int(train_ds.rna_dim),
        omic_sizes=train_ds.omic_sizes or [],
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        combo_mode=combo_mode,
        geo_impl=geo_impl,
        num_points=int(num_points),
        geo_fusion=geo_fusion,
        geo_pool_mode=geo_pool_mode,
        point_feature_dim=int(point_feature_dim),
        plain_omics_pool=str(plain_omics_pool),
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

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)
    print(
        f"[run] combo={combo_mode} geo_impl={geo_impl} K={num_points} fusion={geo_fusion} "
        f"geo_pool={geo_pool_mode} seed={seed} out_dir={out_dir}"
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_risks: list[np.ndarray] = []
        train_times_np: list[np.ndarray] = []
        train_cens_np: list[np.ndarray] = []
        geo_reg_trace: list[float] = []
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_risks.append(risk.detach().cpu().numpy())
            train_times_np.append(np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32))
            train_cens_np.append(np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32))
            geo_reg_trace.append(float((aux_train["smooth_l1"] + aux_train["smooth_l2"]).detach().cpu()))

        train_c_index = concordance_index(
            np.concatenate(train_risks, axis=0),
            np.concatenate(train_times_np, axis=0),
            np.concatenate(train_cens_np, axis=0),
        )
        val_c_index, val_loss = evaluate(model, val_loader, case_table, device)
        test_c_index, _ = evaluate(model, test_loader, case_table, device)

        if val_c_index_ema is None:
            val_c_index_ema = float(val_c_index)
        else:
            decay = float(val_ema_decay)
            val_c_index_ema = decay * float(val_c_index_ema) + (1.0 - decay) * float(val_c_index)

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
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "combo_mode": combo_mode,
                    "geo_impl": geo_impl,
                    "num_points": int(num_points),
                    "geo_fusion": geo_fusion,
                    "geo_pool_mode": geo_pool_mode,
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
            "geo_reg_trace": float(np.mean(geo_reg_trace)) if geo_reg_trace else 0.0,
            "train_c_index": float(train_c_index),
            "val_c_index": float(val_c_index),
            "val_loss": float(val_loss),
            "val_c_index_ema": float(val_c_index_ema),
            "selection_metric": str(selection_metric),
            "selection_score": float(select_score),
            "test_c_index": float(test_c_index),
        }
        history_rows.append(row)
        print(
            f"epoch={epoch} combo={combo_mode} geo_impl={geo_impl} K={num_points} fusion={geo_fusion} "
            f"geo_pool={geo_pool_mode} seed={seed} loss={row['loss']:.4f} "
            f"train_c_index={row['train_c_index']:.4f} val_c_index={row['val_c_index']:.4f} "
            f"test_c_index={row['test_c_index']:.4f}"
        )
        if epoch >= int(selection_min_epochs) and stale_epochs >= int(early_stop_patience):
            print(
                f"early_stop combo={combo_mode} geo_impl={geo_impl} K={num_points} fusion={geo_fusion} "
                f"geo_pool={geo_pool_mode} seed={seed} epoch={epoch} best_epoch={best_epoch}"
            )
            break

    write_csv(
        out_dir / "histories" / f"seed{int(seed)}.csv",
        history_rows,
        ["epoch", "loss", "geo_reg_trace", "train_c_index", "val_c_index", "val_loss", "val_c_index_ema", "selection_metric", "selection_score", "test_c_index"],
    )

    summary = {
        "mode": "rna_geo_survival",
        "seed": int(seed),
        "combo_mode": combo_mode,
        "geo_impl": geo_impl,
        "num_points": int(num_points),
        "geo_fusion": geo_fusion,
        "geo_pool_mode": geo_pool_mode,
        "split_dir": str(split_dir),
        "target_col": str(target_col),
        "rna_tsv_path": str(rna_tsv_path),
        "rna_gene_sets_csv": str(gene_sets_csv),
        "best_epoch": int(best_epoch),
        "best_val_c_index": float(best_val),
        "best_test_c_index": float(best_test),
        "best_train_c_index": float(best_train),
        "best_val_loss": float(best_val_loss),
        "best_select_score": float(best_select_score),
        "n_train_cases": len(train_ds),
        "n_val_cases": len(val_ds),
        "n_test_cases": len(test_ds),
        "out_dir": str(out_dir),
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="RNA geometry sweep: latent points and latent curve descriptors.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--rna_tsv", default=None)
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--combo_modes", default="dense_geo_plus_omics,omics_geo_plus_dense")
    p.add_argument("--geo_impls", default="points,curve")
    p.add_argument("--num_points_list", default="12,24,36")
    p.add_argument("--geo_fusions", default="concat,gated")
    p.add_argument("--geo_pool_modes", default="mean,attn")
    p.add_argument("--plain_omics_pool", default="attn", choices=["mean", "attn"])
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
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--run_tag", default="geo_sweep")
    p.add_argument("--out_dir", default="outputs/rna_geo_branch")
    args = p.parse_args()

    split_dir = Path(args.split_dir).resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"split_dir not found: {split_dir}")
    combo_modes = parse_csv_list(args.combo_modes, default=list(COMBO_CHOICES))
    geo_impls = parse_csv_list(args.geo_impls, default=list(GEO_IMPL_CHOICES))
    num_points_list = [int(x) for x in parse_csv_list(args.num_points_list, default=["12", "24", "36"])]
    geo_fusions = parse_csv_list(args.geo_fusions, default=["concat", "gated"])
    geo_pool_modes = parse_csv_list(args.geo_pool_modes, default=["mean", "attn"])
    seeds = [int(x) for x in parse_csv_list(args.seeds, default=[str(int(args.seed))])]
    if any(x not in COMBO_CHOICES for x in combo_modes):
        raise ValueError(f"unsupported combo_modes: {combo_modes}")
    if any(x not in GEO_IMPL_CHOICES for x in geo_impls):
        raise ValueError(f"unsupported geo_impls: {geo_impls}")
    if any(x not in GEO_FUSION_CHOICES for x in geo_fusions):
        raise ValueError(f"unsupported geo_fusions: {geo_fusions}")
    if any(x not in GEO_POOL_CHOICES for x in geo_pool_modes):
        raise ValueError(f"unsupported geo_pool_modes: {geo_pool_modes}")

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
        for combo_mode in combo_modes:
            for geo_impl in geo_impls:
                for num_points in num_points_list:
                    active_pool_modes = geo_pool_modes if geo_impl == "points" else ["mean"]
                    for geo_pool_mode in active_pool_modes:
                        for geo_fusion in geo_fusions:
                            run_name = (
                                f"combo={combo_mode}__geo={geo_impl}__K={int(num_points)}"
                                f"__fusion={geo_fusion}__pool={geo_pool_mode}"
                            )
                            out_dir = run_root / run_name / f"seed={int(seed)}"
                            summary = train_one_run(
                                split_dir=split_dir,
                                target_col=str(args.target_col),
                                rna_tsv_path=rna_tsv_path,
                                gene_sets_csv=gene_sets_csv,
                                gene_id_to_symbol=gene_id_to_symbol,
                                out_dir=out_dir,
                                seed=int(seed),
                                combo_mode=str(combo_mode),
                                geo_impl=str(geo_impl),
                                num_points=int(num_points),
                                geo_fusion=str(geo_fusion),
                                geo_pool_mode=str(geo_pool_mode),
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
                                smooth_l1_weight=float(args.smooth_l1_weight),
                                smooth_l2_weight=float(args.smooth_l2_weight),
                                curvature_reg_weight=float(args.curvature_reg_weight),
                                torsion_reg_weight=float(args.torsion_reg_weight),
                            )
                            leaderboard_rows.append(summary)
                            leaderboard_fieldnames = [
                                "seed",
                                "combo_mode",
                                "geo_impl",
                                "num_points",
                                "geo_fusion",
                                "geo_pool_mode",
                                "best_epoch",
                                "best_val_c_index",
                                "best_test_c_index",
                                "best_train_c_index",
                                "best_val_loss",
                                "best_select_score",
                                "out_dir",
                            ]
                            leaderboard_export = [
                                {key: row.get(key) for key in leaderboard_fieldnames}
                                for row in sorted(
                                    leaderboard_rows,
                                    key=lambda x: (float(x["best_val_c_index"]), float(x["best_test_c_index"])),
                                    reverse=True,
                                )
                            ]
                            write_csv(run_root / "leaderboard.csv", leaderboard_export, leaderboard_fieldnames)

    save_json(
        run_root / "summary.json",
        {
            "mode": "rna_geo_sweep",
            "split_dir": str(split_dir),
            "run_root": str(run_root),
            "runs": leaderboard_rows,
        },
    )
    print(f"[done] run_root={run_root}")


if __name__ == "__main__":
    main()
