from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(dtype=torch.float32).unsqueeze(-1)
    s = (x * w).sum(dim=1)
    d = w.sum(dim=1).clamp_min(1.0)
    return s / d


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.to(dtype=torch.bool).unsqueeze(-1)
    fill_value = torch.finfo(x.dtype).min
    masked = x.masked_fill(~keep, fill_value)
    out = masked.max(dim=1).values
    valid = mask.sum(dim=1, keepdim=True) > 0
    return torch.where(valid, out, torch.zeros_like(out))


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.to(dtype=torch.bool)
    fill_value = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~keep, fill_value)
    weights = torch.softmax(masked_scores, dim=1)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights / denom


def _build_geo_tokens(tile_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=torch.float32).unsqueeze(-1)
    denom = weight.sum(dim=1).clamp_min(1.0)
    mean_xy = (tile_xy * weight).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
    centered = (tile_xy - mean_xy) * weight
    var_xy = (centered.square().sum(dim=1, keepdim=True) / denom.unsqueeze(1)).clamp_min(1e-6)
    norm_xy = centered / torch.sqrt(var_xy)
    x = norm_xy[..., 0]
    y = norm_xy[..., 1]
    radius = torch.sqrt((x.square() + y.square()).clamp_min(1e-6))
    geo = torch.stack([x, y, radius, x.square(), y.square(), x * y], dim=-1)
    return geo * weight


def _sample_masked_points(tile_xy: torch.Tensor, mask: torch.Tensor, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, coord_dim = tile_xy.shape
    out_points = tile_xy.new_zeros(batch_size, int(num_points), coord_dim)
    out_mask = mask.new_zeros(batch_size, int(num_points))
    for i in range(batch_size):
        valid_idx = torch.nonzero(mask[i].to(dtype=torch.bool), as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() == 1:
            chosen = valid_idx.repeat(int(num_points))
        else:
            lin = torch.linspace(
                0,
                float(valid_idx.numel() - 1),
                steps=int(num_points),
                device=tile_xy.device,
                dtype=torch.float32,
            ).round().long()
            chosen = valid_idx.index_select(0, lin.clamp(0, valid_idx.numel() - 1))
        out_points[i] = tile_xy[i].index_select(0, chosen)
        out_mask[i] = 1
    return out_points, out_mask


def _build_sampled_tile_geo_summary(tile_xy: torch.Tensor, mask: torch.Tensor, num_points: int) -> torch.Tensor:
    sampled_xy, sampled_mask = _sample_masked_points(tile_xy, mask, num_points)
    sampled_geo = _build_geo_tokens(sampled_xy, sampled_mask)
    return _masked_mean(sampled_geo, sampled_mask)


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
    coord_dim = int(points.shape[-1])
    planar = points[:, :, : min(2, coord_dim)]
    radius = torch.sqrt(planar.pow(2).sum(dim=-1) + eps)
    rho = torch.sqrt(points.pow(2).sum(dim=-1) + eps)
    bbox_mean = (points.max(dim=1).values - points.min(dim=1).values).mean(dim=1)
    extra_abs = (
        points[:, :, 2:].abs().mean(dim=(1, 2))
        if coord_dim > 2
        else torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    )
    return torch.stack(
        [
            radius.mean(dim=1),
            radius.std(dim=1, unbiased=False),
            rho.mean(dim=1),
            bbox_mean + 0.1 * extra_abs,
        ],
        dim=1,
    )


def local_geometry(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    coord_dim = int(points.shape[-1])
    r_prime = (points[:, 2:, :] - points[:, :-2, :]) / 2.0
    r_double = points[:, 2:, :] - 2.0 * points[:, 1:-1, :] + points[:, :-2, :]
    norm_rp = torch.norm(r_prime, dim=-1)
    tangent = r_prime / norm_rp.unsqueeze(-1).clamp_min(eps)
    accel_parallel = (r_double * tangent).sum(dim=-1, keepdim=True) * tangent
    accel_perp = r_double - accel_parallel
    curvature = torch.clamp(torch.norm(accel_perp, dim=-1) / (norm_rp.pow(2) + eps), min=0.0)
    if points.shape[1] >= 5:
        r_triple = points[:, 4:, :] - 3.0 * points[:, 3:-1, :] + 3.0 * points[:, 2:-2, :] - points[:, 1:-3, :]
        if coord_dim == 3:
            cross = torch.cross(r_prime, r_double, dim=-1)
            cross_tau = cross[:, 1:-1, :]
            norm_cross_tau = torch.norm(cross_tau, dim=-1)
            torsion = -torch.sum(cross_tau * r_triple, dim=-1) / (norm_cross_tau.pow(2) + eps)
        else:
            tangent_mid = tangent[:, 1:-1, :]
            accel_mid = accel_perp[:, 1:-1, :]
            accel_unit = accel_mid / torch.norm(accel_mid, dim=-1, keepdim=True).clamp_min(eps)
            jerk_res = r_triple
            jerk_res = jerk_res - (jerk_res * tangent_mid).sum(dim=-1, keepdim=True) * tangent_mid
            jerk_res = jerk_res - (jerk_res * accel_unit).sum(dim=-1, keepdim=True) * accel_unit
            torsion = torch.norm(jerk_res, dim=-1) / (norm_rp[:, 1:-1].pow(3) + eps)
    else:
        torsion = torch.zeros(points.shape[0], 1, device=points.device, dtype=points.dtype)
    return curvature, torsion


def pad_geometry_sequence(values: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = int(values.shape[1])
    if current_len == target_len:
        return values
    if current_len <= 0:
        return torch.zeros(values.shape[0], target_len, device=values.device, dtype=values.dtype)
    if current_len > target_len:
        return values[:, :target_len]
    deficit = target_len - current_len
    left = deficit // 2
    right = deficit - left
    pieces: list[torch.Tensor] = []
    if left > 0:
        pieces.append(values[:, :1].repeat(1, left))
    pieces.append(values)
    if right > 0:
        pieces.append(values[:, -1:].repeat(1, right))
    return torch.cat(pieces, dim=1)


def geometry_per_point_tensor(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    curvature, torsion = local_geometry(points)
    num_points = int(points.shape[1])
    curvature_full = pad_geometry_sequence(curvature, num_points)
    torsion_full = pad_geometry_sequence(torsion, num_points)
    geometry_per_point = torch.stack([curvature_full, torsion_full], dim=-1)
    return curvature_full, torsion_full, geometry_per_point


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


def summarize_enhanced_geometry_stats(points: torch.Tensor, curvature: torch.Tensor, torsion: torch.Tensor) -> torch.Tensor:
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
    return torch.cat(
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


class CurveGeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, num_points: int, coord_dim: int = 3) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.coord_dim = int(coord_dim)
        self.delta_proj = nn.Linear(int(input_dim), self.num_points * self.coord_dim)
        self.anchor_proj = nn.Linear(int(input_dim), self.coord_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = x.shape[0]
        deltas = self.delta_proj(x).view(bsz, self.num_points, self.coord_dim)
        points = torch.cumsum(deltas, dim=1) + self.anchor_proj(x).unsqueeze(1)
        curvature, torsion = local_geometry(points)
        curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
        geometry_enhanced = summarize_enhanced_geometry_stats(points, curvature, torsion)
        return {
            "points": points,
            "curvature": curvature,
            "torsion": torsion,
            "curvature_per_point": curvature_full,
            "torsion_per_point": torsion_full,
            "geometry_per_point": geometry_per_point,
            "geometry_enhanced": geometry_enhanced,
        }


class TokenPointGeometryEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_points: int, coord_dim: int = 3) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.coord_dim = int(coord_dim)
        self.query = nn.Parameter(torch.randn(self.num_points, int(hidden_dim)) * 0.02)
        self.key_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.value_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.init_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.gru = nn.GRUCell(int(hidden_dim), int(hidden_dim))
        self.delta_proj = nn.Linear(int(hidden_dim), self.coord_dim)
        self.anchor_proj = nn.Linear(int(hidden_dim), self.coord_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, *, return_attn: bool = True) -> dict[str, torch.Tensor]:
        batch_size, _, hidden_dim = tokens.shape
        keep = mask.to(dtype=torch.bool)
        has_any = keep.any(dim=1)

        pooled = _masked_mean(tokens, mask)
        h = torch.tanh(self.init_proj(pooled))

        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.einsum("bkd,btd->bkt", queries, keys) / (float(hidden_dim) ** 0.5)
        fill_value = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~keep.unsqueeze(1), fill_value)
        masked_scores = masked_scores.clone()
        masked_scores[~has_any] = 0.0
        weights = torch.softmax(masked_scores, dim=-1)
        weights = weights * keep.to(dtype=weights.dtype).unsqueeze(1)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        weights = torch.where(has_any.view(batch_size, 1, 1), weights, torch.zeros_like(weights))
        context = torch.einsum("bkt,btd->bkd", weights, values)

        anchor = self.anchor_proj(pooled).unsqueeze(1)
        deltas: list[torch.Tensor] = []
        for idx in range(self.num_points):
            h = self.gru(context[:, idx, :], h)
            deltas.append(self.delta_proj(h))
        delta_tensor = torch.stack(deltas, dim=1)
        points = torch.cumsum(delta_tensor, dim=1) + anchor

        curvature, torsion = local_geometry(points)
        curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
        geometry_enhanced = summarize_enhanced_geometry_stats(points, curvature, torsion)
        out = {
            "points": points,
            "curvature": curvature,
            "torsion": torsion,
            "curvature_per_point": curvature_full,
            "torsion_per_point": torsion_full,
            "geometry_per_point": geometry_per_point,
            "geometry_enhanced": geometry_enhanced,
        }
        if return_attn:
            out["token_attn"] = weights
        return out


def _make_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(out_dim)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
    )


class OmicsMLPEncoder(nn.Module):
    def __init__(self, omic_sizes: list[int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.sig_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(s), int(hidden_dim)),
                    nn.ELU(),
                    nn.AlphaDropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.ELU(),
                    nn.AlphaDropout(float(dropout)),
                )
                for s in omic_sizes
            ]
        )
        self.attn_score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, rna_omics: list[torch.Tensor], pool_mode: str = "mean") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(rna_omics) != len(self.sig_networks):
            raise ValueError("rna_omics length mismatch")
        feats = [net(x.float()) for net, x in zip(self.sig_networks, rna_omics, strict=True)]
        tokens = torch.stack(feats, dim=1)
        if str(pool_mode) == "attn":
            attn_logits = self.attn_score(tokens).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            case = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        else:
            attn_weights = torch.full(
                (tokens.shape[0], tokens.shape[1]),
                1.0 / float(tokens.shape[1]),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            case = tokens.mean(dim=1)
        return case, tokens, attn_weights


class SlideLevelAggregator(nn.Module):
    """Pool tiles to slides, then pool slides to a case embedding."""

    def __init__(self, hidden_dim: int, mode: str = "slide_mean_case_attn") -> None:
        super().__init__()
        if mode not in {"slide_mean_case_attn", "slide_attn_case_attn"}:
            raise ValueError("multi-slide mode must be slide_mean_case_attn|slide_attn_case_attn")
        self.mode = str(mode)
        self.tile_attn = nn.Linear(int(hidden_dim), 1)
        self.slide_pool = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.slide_attn = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        tile_features: torch.Tensor,
        slide_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, _, hidden_dim = tile_features.shape
        valid_mask = mask.to(dtype=torch.bool)
        valid_slide_ids = slide_ids.masked_fill(~valid_mask, -1)
        max_slide_count = int(valid_slide_ids.max().item()) + 1 if bool(valid_mask.any()) else 1

        slide_features = tile_features.new_zeros((batch_size, max_slide_count, hidden_dim))
        slide_valid = torch.zeros((batch_size, max_slide_count), dtype=torch.bool, device=tile_features.device)
        tile_attn_by_slide = (
            tile_features.new_zeros((batch_size, max_slide_count, tile_features.shape[1]))
            if self.mode == "slide_attn_case_attn"
            else None
        )
        tile_scores = self.tile_attn(tile_features).squeeze(-1) if self.mode == "slide_attn_case_attn" else None

        for slide_idx in range(max_slide_count):
            slide_mask = (valid_slide_ids == slide_idx) & valid_mask
            if not bool(slide_mask.any()):
                continue
            slide_mask_f = slide_mask.to(dtype=mask.dtype)
            if self.mode == "slide_attn_case_attn":
                if tile_scores is None:
                    raise RuntimeError("tile attention scores are required for slide_attn_case_attn")
                local_weights = _masked_softmax(tile_scores, slide_mask_f)
                slide_features[:, slide_idx, :] = (tile_features * local_weights.unsqueeze(-1)).sum(dim=1)
                if tile_attn_by_slide is not None:
                    tile_attn_by_slide[:, slide_idx, :] = local_weights
            else:
                slide_features[:, slide_idx, :] = _masked_mean(tile_features, slide_mask_f)
            slide_valid[:, slide_idx] = slide_mask.any(dim=1)

        slide_hidden = self.slide_pool(slide_features)
        attn_scores = self.slide_attn(slide_hidden).squeeze(-1)
        attn_weights = _masked_softmax(attn_scores, slide_valid.to(dtype=mask.dtype))
        case_feat = (attn_weights.unsqueeze(-1) * slide_hidden).sum(dim=1)
        return case_feat, attn_weights, slide_hidden, tile_attn_by_slide


class DirectWSIRNASurvival(nn.Module):
    def __init__(
        self,
        *,
        tile_dim: int = 512,
        rna_dim: int = 19962,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        model_variant: str = "baseline",
        aug_dim: int = 64,
        geo_dim: int = 64,
        pool_method: str = "attention",
        gate_enabled: bool | None = None,
        wsi_geo_type: str = "none",
        wsi_geo_num_points: int = 24,
        wsi_geo_coord_dim: int = 3,
        wsi_geo_output: str = "points",
        wsi_geo_position: str = "after_mean",
        wsi_geo_fusion: str = "concat",
        wsi_b_points_level: str = "case",
        rna_geo_type: str = "none",
        rna_geo_num_points: int = 6,
        rna_geo_coord_dim: int = 3,
        rna_geo_output: str = "inherit",
        rna_geo_position: str = "after_rna_proj",
        rna_geo_fusion: str = "concat",
        geo_swap: bool = False,
        use_multi_slide: bool = False,
        multi_slide_mode: str = "slide_mean_case_attn",
        rna_mode: str = "vec",
        rna_omic_sizes: list[int] | None = None,
        cross_modal_fusion: str = "concat",
        anti_dim: int = 128,
        use_anti_injection: bool = False,
    ) -> None:
        super().__init__()
        self.use_anti_injection = bool(use_anti_injection)
        self.anti_dim = int(anti_dim) if self.use_anti_injection else 0
        valid_variants = {
            "baseline",
            "gate_only",
            "aug_only",
            "geo_only",
            "geo_gate",
            "geo_gate_v2",
            "geo_tile_attn",
            "geo_tile",
            "custom",
        }
        if model_variant not in valid_variants:
            raise ValueError("unsupported model_variant")
        if pool_method not in {"mean", "max", "attention", "gated_attention", "attention_sa"}:
            raise ValueError("pool_method must be mean|max|attention|gated_attention|attention_sa")
        if wsi_geo_type not in {"none", "curve", "b_points"}:
            raise ValueError("wsi_geo_type must be none|curve|b_points")
        if wsi_geo_output not in {"points", "curvature", "enhanced", "all"}:
            raise ValueError("wsi_geo_output must be points|curvature|enhanced|all")
        if rna_geo_output not in {"inherit", "points", "curvature", "enhanced", "all"}:
            raise ValueError("rna_geo_output must be inherit|points|curvature|enhanced|all")
        if wsi_geo_type == "b_points" and wsi_geo_output in {"curvature", "enhanced"}:
            raise ValueError("b_points requires wsi_geo_output to include points (points|all)")
        if wsi_geo_position not in {"after_mean", "before_mean", "parallel"}:
            raise ValueError("wsi_geo_position must be after_mean|before_mean|parallel")
        if wsi_geo_fusion not in {"concat", "gated"}:
            raise ValueError("wsi_geo_fusion must be concat|gated")
        if wsi_b_points_level not in {"case", "slide"}:
            raise ValueError("wsi_b_points_level must be case|slide")
        if rna_geo_type not in {"none", "tile", "curve"}:
            raise ValueError("rna_geo_type must be none|tile|curve")
        if rna_mode not in {"vec", "omics", "omics_attn"}:
            raise ValueError("rna_mode must be vec|omics|omics_attn")
        if cross_modal_fusion not in {
            "concat",
            "gated",
            "self_attn",
            "cross_attn_rna_queries_wsi",
            "cross_attn_wsi_queries_rna",
            "cross_attn_bidirectional",
            "cross_attn_self_attn",
            "across",
        }:
            raise ValueError(
                "cross_modal_fusion must be concat|gated|self_attn|"
                "cross_attn_rna_queries_wsi|cross_attn_wsi_queries_rna|"
                "cross_attn_bidirectional|cross_attn_self_attn|across"
            )
        if rna_geo_position not in {"after_rna_proj", "before_gate", "at_fused"}:
            raise ValueError("rna_geo_position must be after_rna_proj|before_gate|at_fused")
        if rna_geo_fusion not in {"concat", "multiply", "add"}:
            raise ValueError("rna_geo_fusion must be concat|multiply|add")
        if int(wsi_geo_num_points) <= 0 or int(rna_geo_num_points) <= 0:
            raise ValueError("geo num_points must be positive")
        if int(wsi_geo_coord_dim) not in {2, 3, 4} or int(rna_geo_coord_dim) not in {2, 3, 4}:
            raise ValueError("geo coord dim must be one of {2,3,4}")

        self.model_variant = str(model_variant)
        self.hidden_dim = int(hidden_dim)
        self.aug_dim = int(aug_dim)
        self.geo_dim = int(geo_dim)
        self.pool_method = str(pool_method)
        self.use_custom = self.model_variant == "custom"
        self.use_gate = self._resolve_gate_enabled(gate_enabled)
        self.use_multi_slide = bool(use_multi_slide)
        self.multi_slide_mode = str(multi_slide_mode)
        self.rna_mode = str(rna_mode)
        self.cross_modal_fusion = str(cross_modal_fusion)
        if not self.use_custom:
            if str(rna_geo_type) != "none":
                raise ValueError("rna_geo_type is only executed when model_variant=custom")
            if str(wsi_geo_type) == "curve":
                raise ValueError("wsi_geo_type=curve is only executed when model_variant=custom")
        if not self.use_custom and str(wsi_geo_type) == "b_points" and str(wsi_geo_position) == "parallel":
            raise ValueError("wsi_geo_position=parallel with b_points is only supported when model_variant=custom")
        if str(wsi_geo_type) == "b_points" and str(wsi_geo_position) == "before_mean" and str(wsi_b_points_level) == "slide":
            raise ValueError("b_points slide-level generation is not supported when wsi_geo_position=before_mean")

        self.wsi_proj = _make_mlp(int(tile_dim), int(hidden_dim), float(dropout))
        self.rna_proj = _make_mlp(int(rna_dim), int(hidden_dim), float(dropout)) if self.rna_mode == "vec" else None
        if self.rna_mode in {"omics", "omics_attn"}:
            if not rna_omic_sizes:
                raise ValueError("rna_omic_sizes are required when rna_mode uses omics tokens")
            self.rna_omics_encoder = OmicsMLPEncoder(list(rna_omic_sizes), int(hidden_dim), float(dropout))
        else:
            self.rna_omics_encoder = None

        self.use_aug = self.model_variant == "aug_only"
        self.use_geo = self.model_variant in {"geo_only", "geo_gate", "geo_gate_v2", "geo_tile_attn", "geo_tile"}
        self.use_geo_gate_v2 = self.model_variant == "geo_gate_v2"
        self.use_geo_tile = self.model_variant in {"geo_tile_attn", "geo_tile"}

        if self.use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Sigmoid(),
            )
        else:
            self.gate_net = None

        if self.use_aug:
            self.aug_proj = _make_mlp(int(hidden_dim) * 2, int(self.aug_dim), float(dropout))
            self.aug_align = _make_mlp(int(hidden_dim) + int(self.aug_dim), int(hidden_dim), float(dropout))
        else:
            self.aug_proj = None
            self.aug_align = None

        if self.use_geo:
            self.geo_proj = _make_mlp(6, int(self.geo_dim), float(dropout))
            if self.use_geo_tile:
                self.geo_align = None
                self.geo_mix_gate = None
                self.tile_fusion_proj = _make_mlp(int(hidden_dim) + int(self.geo_dim), int(hidden_dim), float(dropout))
            elif self.use_geo_gate_v2:
                self.geo_align = _make_mlp(int(self.geo_dim), int(hidden_dim), float(dropout))
                self.geo_mix_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
                self.tile_fusion_proj = None
            else:
                self.geo_align = _make_mlp(int(hidden_dim) + int(self.geo_dim), int(hidden_dim), float(dropout))
                self.geo_mix_gate = None
                self.tile_fusion_proj = None
        else:
            self.geo_proj = None
            self.geo_align = None
            self.geo_mix_gate = None
            self.tile_fusion_proj = None

        if self.pool_method == "attention":
            self.tile_attention_score = nn.Linear(int(hidden_dim), 1)
            self.tile_attention_v = None
            self.tile_attention_u = None
            self.tile_self_attn = None
        elif self.pool_method == "gated_attention":
            self.tile_attention_score = nn.Linear(int(hidden_dim), 1)
            self.tile_attention_v = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.tile_attention_u = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.tile_self_attn = None
        elif self.pool_method == "attention_sa":
            self.tile_attention_score = nn.Linear(int(hidden_dim), 1)
            self.tile_attention_v = None
            self.tile_attention_u = None
            self.tile_self_attn = nn.MultiheadAttention(
                embed_dim=int(hidden_dim),
                num_heads=4,
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.tile_attention_score = None
            self.tile_attention_v = None
            self.tile_attention_u = None
            self.tile_self_attn = None

        self.slide_aggregator = (
            SlideLevelAggregator(int(hidden_dim), mode=self.multi_slide_mode) if self.use_multi_slide else None
        )

        self.cross_modal_gate = None
        self.cross_modal_self_attn = None
        self.cross_modal_rna_queries_wsi_attn = None
        self.cross_modal_wsi_queries_rna_attn = None
        self.cross_modal_rna_update = None
        self.cross_modal_wsi_update = None
        self.cross_modal_across_proj = None
        if self.cross_modal_fusion == "gated":
            self.cross_modal_gate = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.Sigmoid(),
            )
        if self.cross_modal_fusion in {"self_attn", "cross_attn_self_attn"}:
            self.cross_modal_self_attn = nn.MultiheadAttention(
                embed_dim=int(hidden_dim),
                num_heads=4,
                dropout=float(dropout),
                batch_first=True,
            )
        if self.cross_modal_fusion in {
            "cross_attn_rna_queries_wsi",
            "cross_attn_bidirectional",
            "cross_attn_self_attn",
        }:
            self.cross_modal_rna_queries_wsi_attn = nn.MultiheadAttention(
                embed_dim=int(hidden_dim),
                num_heads=4,
                dropout=float(dropout),
                batch_first=True,
            )
            self.cross_modal_rna_update = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
        if self.cross_modal_fusion in {
            "cross_attn_wsi_queries_rna",
            "cross_attn_bidirectional",
            "cross_attn_self_attn",
        }:
            self.cross_modal_wsi_queries_rna_attn = nn.MultiheadAttention(
                embed_dim=int(hidden_dim),
                num_heads=4,
                dropout=float(dropout),
                batch_first=True,
            )
            self.cross_modal_wsi_update = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
        if self.cross_modal_fusion == "across":
            self.cross_modal_across_proj = _make_mlp(int(hidden_dim) * 4, int(hidden_dim) * 2, float(dropout))

        self.wsi_geo_type = str(wsi_geo_type)
        self.wsi_geo_num_points = int(wsi_geo_num_points)
        self.wsi_geo_coord_dim = int(wsi_geo_coord_dim)
        self.wsi_geo_output = str(wsi_geo_output)
        self.wsi_geo_position = str(wsi_geo_position)
        self.wsi_geo_fusion = str(wsi_geo_fusion)
        self.wsi_b_points_level = str(wsi_b_points_level)
        self.rna_geo_type = str(rna_geo_type)
        self.rna_geo_num_points = int(rna_geo_num_points)
        self.rna_geo_coord_dim = int(rna_geo_coord_dim)
        self.rna_geo_output = str(rna_geo_output)
        self.rna_geo_position = str(rna_geo_position)
        self.rna_geo_fusion = str(rna_geo_fusion)
        self.geo_swap = bool(geo_swap)

        self.effective_rna_geo_output = self.wsi_geo_output if self.rna_geo_output == "inherit" else self.rna_geo_output

        self.effective_wsi_geo_type = self.rna_geo_type if self.geo_swap else self.wsi_geo_type
        self.effective_rna_geo_type = self.wsi_geo_type if self.geo_swap else self.rna_geo_type
        self.effective_wsi_geo_num_points = self.rna_geo_num_points if self.geo_swap else self.wsi_geo_num_points
        self.effective_rna_geo_num_points = self.wsi_geo_num_points if self.geo_swap else self.rna_geo_num_points
        self.effective_wsi_geo_coord_dim = self.rna_geo_coord_dim if self.geo_swap else self.wsi_geo_coord_dim
        self.effective_rna_geo_coord_dim = self.wsi_geo_coord_dim if self.geo_swap else self.rna_geo_coord_dim

        self.wsi_curve_encoder = None
        self.rna_curve_encoder = None
        self.wsi_b_points_encoder = None
        self.wsi_geo_feature_proj = None
        self.rna_geo_feature_proj = None
        self.wsi_geo_after_align = None
        self.wsi_geo_after_gate = None
        self.wsi_geo_before_align = None
        self.wsi_geo_before_gate = None
        self.wsi_geo_parallel_gate = None
        self.rna_geo_after_align = None
        self.rna_geo_before_gate_align = None
        self.rna_geo_at_fused_align = None

        custom_fused_dim = int(hidden_dim) * 2
        if self.use_custom:
            if self.effective_wsi_geo_type == "curve":
                self.wsi_curve_encoder = CurveGeometryEncoder(
                    int(hidden_dim),
                    self.effective_wsi_geo_num_points,
                    self.effective_wsi_geo_coord_dim,
                )
                self.wsi_geo_feature_proj = _make_mlp(
                    self._curve_output_dim(
                        self.wsi_geo_output,
                        self.effective_wsi_geo_num_points,
                        self.effective_wsi_geo_coord_dim,
                    ),
                    int(hidden_dim),
                    float(dropout),
                )
            elif self.effective_wsi_geo_type == "b_points":
                self.wsi_b_points_encoder = TokenPointGeometryEncoder(
                    int(hidden_dim),
                    self.effective_wsi_geo_num_points,
                    self.effective_wsi_geo_coord_dim,
                )
                self.wsi_geo_feature_proj = _make_mlp(
                    self._curve_output_dim(
                        self.wsi_geo_output,
                        self.effective_wsi_geo_num_points,
                        self.effective_wsi_geo_coord_dim,
                    ),
                    int(hidden_dim),
                    float(dropout),
                )
            elif self.effective_wsi_geo_type == "tile":
                self.wsi_geo_feature_proj = _make_mlp(6, int(hidden_dim), float(dropout))

            if self.effective_rna_geo_type == "curve":
                self.rna_curve_encoder = CurveGeometryEncoder(
                    int(hidden_dim),
                    self.effective_rna_geo_num_points,
                    self.effective_rna_geo_coord_dim,
                )
                self.rna_geo_feature_proj = _make_mlp(
                    self._curve_output_dim(
                        self.effective_rna_geo_output,
                        self.effective_rna_geo_num_points,
                        self.effective_rna_geo_coord_dim,
                    ),
                    int(hidden_dim),
                    float(dropout),
                )
            elif self.effective_rna_geo_type == "tile":
                self.rna_geo_feature_proj = _make_mlp(6, int(hidden_dim), float(dropout))

            if self.effective_wsi_geo_type != "none":
                if self.wsi_geo_position == "after_mean":
                    if self.wsi_geo_fusion == "concat":
                        self.wsi_geo_after_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                    else:
                        self.wsi_geo_after_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
                elif self.wsi_geo_position == "before_mean":
                    if self.wsi_geo_fusion == "concat":
                        self.wsi_geo_before_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                    else:
                        self.wsi_geo_before_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
                elif self.wsi_geo_position == "parallel" and self.wsi_geo_fusion == "gated":
                    self.wsi_geo_parallel_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
                if self.wsi_geo_position == "parallel":
                    custom_fused_dim += int(hidden_dim)

            if self.effective_rna_geo_type != "none":
                if self.rna_geo_position == "after_rna_proj" and self.rna_geo_fusion == "concat":
                    self.rna_geo_after_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                if self.rna_geo_position == "before_gate" and self.rna_geo_fusion == "concat":
                    self.rna_geo_before_gate_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                if self.rna_geo_position == "at_fused":
                    if self.rna_geo_fusion == "concat":
                        self.rna_geo_at_fused_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                    custom_fused_dim += int(hidden_dim)

        if (not self.use_custom) and self.effective_wsi_geo_type == "b_points":
            self.wsi_b_points_encoder = TokenPointGeometryEncoder(
                int(hidden_dim),
                self.effective_wsi_geo_num_points,
                self.effective_wsi_geo_coord_dim,
            )
            self.wsi_geo_feature_proj = _make_mlp(
                self._curve_output_dim(
                    self.wsi_geo_output,
                    self.effective_wsi_geo_num_points,
                    self.effective_wsi_geo_coord_dim,
                ),
                int(hidden_dim),
                float(dropout),
            )
            if self.wsi_geo_position == "after_mean":
                if self.wsi_geo_fusion == "concat":
                    self.wsi_geo_after_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                else:
                    self.wsi_geo_after_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
            elif self.wsi_geo_position == "before_mean":
                if self.wsi_geo_fusion == "concat":
                    self.wsi_geo_before_align = _make_mlp(int(hidden_dim) * 2, int(hidden_dim), float(dropout))
                else:
                    self.wsi_geo_before_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())

        self.backbone = nn.Sequential(
            nn.Linear(custom_fused_dim if self.use_custom else int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(int(hidden_dim), 1)

        if self.use_anti_injection:
            self.wsi_anti_proj = nn.Linear(self.anti_dim, int(hidden_dim), bias=False)
            self.wsi_anti_gate = nn.Sequential(
                nn.Linear(int(hidden_dim) + self.anti_dim, int(hidden_dim)),
                nn.Sigmoid(),
            )
            self.rna_anti_proj = nn.Linear(self.anti_dim, int(hidden_dim), bias=False)
            self.rna_anti_gate = nn.Sequential(
                nn.Linear(int(hidden_dim) + self.anti_dim, int(hidden_dim)),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.wsi_anti_proj.weight)
            nn.init.zeros_(self.rna_anti_proj.weight)
        else:
            self.wsi_anti_proj = None
            self.wsi_anti_gate = None
            self.rna_anti_proj = None
            self.rna_anti_gate = None

    def _apply_anti_injection(
        self,
        case: torch.Tensor,
        anti_vec: torch.Tensor | None,
        *,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_anti_injection or anti_vec is None:
            return case, None
        if branch == "wsi":
            proj = self.wsi_anti_proj
            gate_net = self.wsi_anti_gate
        elif branch == "rna":
            proj = self.rna_anti_proj
            gate_net = self.rna_anti_gate
        else:
            raise ValueError(f"unknown branch: {branch}")
        if proj is None or gate_net is None:
            return case, None
        anti_vec = anti_vec.to(dtype=case.dtype, device=case.device)
        proj_anti = proj(anti_vec)
        gate = gate_net(torch.cat([case, anti_vec], dim=1))
        out = case + gate * proj_anti
        return out, gate

    def _resolve_gate_enabled(self, gate_enabled: bool | None) -> bool:
        if gate_enabled is not None:
            return bool(gate_enabled)
        return self.model_variant in {"gate_only", "geo_gate", "geo_gate_v2"}

    def _curve_output_dim(self, geo_output: str, num_points: int, coord_dim: int) -> int:
        if str(geo_output) == "points":
            return int(num_points) * int(coord_dim)
        if str(geo_output) == "curvature":
            return int(num_points) * 2
        if str(geo_output) == "enhanced":
            return 23
        return int(num_points) * (int(coord_dim) + 2) + 23

    def _extract_curve_feature(self, geometry: dict[str, torch.Tensor], *, geo_output: str) -> torch.Tensor:
        if str(geo_output) == "points":
            return geometry["points"].reshape(geometry["points"].shape[0], -1)
        if str(geo_output) == "curvature":
            return geometry["geometry_per_point"].reshape(geometry["geometry_per_point"].shape[0], -1)
        if str(geo_output) == "enhanced":
            return geometry["geometry_enhanced"]
        return torch.cat(
            [
                geometry["points"].reshape(geometry["points"].shape[0], -1),
                geometry["geometry_per_point"].reshape(geometry["geometry_per_point"].shape[0], -1),
                geometry["geometry_enhanced"],
            ],
            dim=1,
        )

    def _pool_wsi_tokens(self, wsi_tokens: torch.Tensor, tile_attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pool_method == "mean":
            return _masked_mean(wsi_tokens, tile_attn_mask), None
        if self.pool_method == "max":
            return _masked_max(wsi_tokens, tile_attn_mask), None
        pooled_tokens = wsi_tokens
        if self.pool_method == "attention_sa":
            if self.tile_self_attn is None or self.tile_attention_score is None:
                raise RuntimeError("attention_sa requires self-attention and attention score layers")
            pooled_tokens, _ = self.tile_self_attn(
                wsi_tokens,
                wsi_tokens,
                wsi_tokens,
                key_padding_mask=~tile_attn_mask.to(dtype=torch.bool),
                need_weights=False,
            )
            scores = self.tile_attention_score(pooled_tokens).squeeze(-1)
        elif self.pool_method == "attention":
            if self.tile_attention_score is None:
                raise RuntimeError("attention pooling requires attention score layer")
            scores = self.tile_attention_score(wsi_tokens).squeeze(-1)
        elif self.pool_method == "gated_attention":
            if self.tile_attention_score is None or self.tile_attention_v is None or self.tile_attention_u is None:
                raise RuntimeError("gated_attention requires ABMIL gating layers")
            gated_hidden = torch.tanh(self.tile_attention_v(wsi_tokens)) * torch.sigmoid(self.tile_attention_u(wsi_tokens))
            scores = self.tile_attention_score(gated_hidden).squeeze(-1)
        else:
            raise RuntimeError(f"Unsupported pool_method: {self.pool_method}")
        attn_weights = _masked_softmax(scores, tile_attn_mask)
        pooled = (pooled_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        return pooled, attn_weights

    def _custom_curve_feature(
        self,
        encoder: CurveGeometryEncoder | None,
        proj: nn.Module | None,
        source: torch.Tensor,
        *,
        geo_output: str,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if encoder is None or proj is None:
            return None, {}
        geometry = encoder(source)
        feature = proj(self._extract_curve_feature(geometry, geo_output=str(geo_output)))
        aux = {f"curve_{k}": v for k, v in geometry.items()}
        return feature, aux

    def _custom_b_points_feature(
        self,
        encoder: TokenPointGeometryEncoder | None,
        proj: nn.Module | None,
        tokens: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        *,
        geo_output: str,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if encoder is None or proj is None:
            return None, {}
        geometry = encoder(tokens, tile_attn_mask)
        feature = proj(self._extract_curve_feature(geometry, geo_output=str(geo_output)))
        aux = {f"b_points_{k}": v for k, v in geometry.items()}
        return feature, aux

    def _custom_tile_feature(
        self,
        proj: nn.Module | None,
        tile_xy: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        num_points: int,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if proj is None:
            return None, {}
        summary = _build_sampled_tile_geo_summary(tile_xy, tile_attn_mask, int(num_points))
        return proj(summary), {"tile_geo_summary": summary}

    def _apply_rna_geo_fusion(self, base: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        if self.rna_geo_fusion == "concat":
            if self.rna_geo_after_align is not None:
                return self.rna_geo_after_align(torch.cat([base, geo], dim=1))
            if self.rna_geo_before_gate_align is not None:
                return self.rna_geo_before_gate_align(torch.cat([base, geo], dim=1))
            if self.rna_geo_at_fused_align is not None:
                return self.rna_geo_at_fused_align(torch.cat([base, geo], dim=1))
            raise RuntimeError("concat fusion requires an alignment layer")
        if self.rna_geo_fusion == "add":
            return base + geo
        if self.rna_geo_fusion == "multiply":
            return base * torch.sigmoid(geo)
        raise RuntimeError(f"unsupported rna_geo_fusion: {self.rna_geo_fusion}")

    def _encode_rna(
        self,
        *,
        rna_vec: torch.Tensor | None,
        rna_omics: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.rna_mode == "vec":
            if rna_vec is None or self.rna_proj is None:
                raise ValueError("rna_vec input is required when rna_mode=vec")
            rna_case = self.rna_proj(rna_vec)
            aux["rna_dense_case"] = rna_case
            return rna_case, aux
        if self.rna_omics_encoder is None:
            raise RuntimeError("omics encoder is not initialized")
        if rna_omics is None:
            raise ValueError("rna_omics input is required when rna_mode uses omics tokens")
        pool_mode = "attn" if self.rna_mode == "omics_attn" else "mean"
        rna_case, omics_tokens, omics_attn_weights = self.rna_omics_encoder(rna_omics, pool_mode=pool_mode)
        aux["rna_omics_case"] = rna_case
        aux["rna_omics_tokens"] = omics_tokens
        aux["rna_omics_attn_weights"] = omics_attn_weights
        return rna_case, aux

    def _cross_modal_update(
        self,
        *,
        query_case: torch.Tensor,
        context_case: torch.Tensor,
        attn_layer: nn.MultiheadAttention | None,
        update_layer: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attn_layer is None or update_layer is None:
            raise RuntimeError("cross-modal attention update requires attention and update layers")
        attended, attn_weights = attn_layer(
            query_case.unsqueeze(1),
            context_case.unsqueeze(1),
            context_case.unsqueeze(1),
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended.squeeze(1)
        updated = update_layer(torch.cat([query_case, attended], dim=1))
        return updated, attn_weights

    def _fuse_modalities(
        self,
        *,
        wsi_case: torch.Tensor,
        rna_case: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {"cross_modal_fusion": torch.empty(0, device=wsi_case.device)}
        if self.cross_modal_fusion == "concat":
            return torch.cat([wsi_case, rna_case], dim=1), aux
        if self.cross_modal_fusion == "gated":
            if self.cross_modal_gate is None:
                raise RuntimeError("gated cross-modal fusion requires cross_modal_gate")
            gate = self.cross_modal_gate(torch.cat([wsi_case, rna_case], dim=1))
            aux["cross_modal_gate"] = gate
            return torch.cat([gate * wsi_case, (1.0 - gate) * rna_case], dim=1), aux
        if self.cross_modal_fusion == "self_attn":
            if self.cross_modal_self_attn is None:
                raise RuntimeError("self_attn cross-modal fusion requires cross_modal_self_attn")
            tokens = torch.stack([wsi_case, rna_case], dim=1)
            fused_tokens, attn_weights = self.cross_modal_self_attn(tokens, tokens, tokens, need_weights=True)
            aux["cross_modal_self_attn_weights"] = attn_weights
            return fused_tokens.reshape(fused_tokens.shape[0], -1), aux
        if self.cross_modal_fusion == "cross_attn_rna_queries_wsi":
            updated_rna, attn_weights = self._cross_modal_update(
                query_case=rna_case,
                context_case=wsi_case,
                attn_layer=self.cross_modal_rna_queries_wsi_attn,
                update_layer=self.cross_modal_rna_update,
            )
            aux["cross_modal_rna_queries_wsi_attn_weights"] = attn_weights
            aux["cross_modal_rna_case_updated"] = updated_rna
            return torch.cat([wsi_case, updated_rna], dim=1), aux
        if self.cross_modal_fusion == "cross_attn_wsi_queries_rna":
            updated_wsi, attn_weights = self._cross_modal_update(
                query_case=wsi_case,
                context_case=rna_case,
                attn_layer=self.cross_modal_wsi_queries_rna_attn,
                update_layer=self.cross_modal_wsi_update,
            )
            aux["cross_modal_wsi_queries_rna_attn_weights"] = attn_weights
            aux["cross_modal_wsi_case_updated"] = updated_wsi
            return torch.cat([updated_wsi, rna_case], dim=1), aux
        if self.cross_modal_fusion in {"cross_attn_bidirectional", "cross_attn_self_attn"}:
            updated_rna, rna_attn = self._cross_modal_update(
                query_case=rna_case,
                context_case=wsi_case,
                attn_layer=self.cross_modal_rna_queries_wsi_attn,
                update_layer=self.cross_modal_rna_update,
            )
            updated_wsi, wsi_attn = self._cross_modal_update(
                query_case=wsi_case,
                context_case=rna_case,
                attn_layer=self.cross_modal_wsi_queries_rna_attn,
                update_layer=self.cross_modal_wsi_update,
            )
            aux["cross_modal_rna_queries_wsi_attn_weights"] = rna_attn
            aux["cross_modal_wsi_queries_rna_attn_weights"] = wsi_attn
            aux["cross_modal_rna_case_updated"] = updated_rna
            aux["cross_modal_wsi_case_updated"] = updated_wsi
            if self.cross_modal_fusion == "cross_attn_bidirectional":
                return torch.cat([updated_wsi, updated_rna], dim=1), aux
            if self.cross_modal_self_attn is None:
                raise RuntimeError("cross_attn_self_attn requires cross_modal_self_attn")
            tokens = torch.stack([updated_wsi, updated_rna], dim=1)
            fused_tokens, attn_weights = self.cross_modal_self_attn(tokens, tokens, tokens, need_weights=True)
            aux["cross_modal_self_attn_weights"] = attn_weights
            return fused_tokens.reshape(fused_tokens.shape[0], -1), aux
        if self.cross_modal_fusion == "across":
            if self.cross_modal_across_proj is None:
                raise RuntimeError("across fusion requires cross_modal_across_proj")
            interaction = torch.cat([wsi_case, rna_case, wsi_case * rna_case, (wsi_case - rna_case).abs()], dim=1)
            aux["cross_modal_interaction"] = interaction
            return self.cross_modal_across_proj(interaction), aux
        raise RuntimeError(f"unsupported cross_modal_fusion: {self.cross_modal_fusion}")

    def _pool_multislide_tokens(
        self,
        wsi_tokens: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        slide_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.use_multi_slide:
            if slide_ids is None or self.slide_aggregator is None:
                raise RuntimeError("multi-slide mode requires slide_ids and slide_aggregator")
            wsi_case, slide_attn_weights, slide_hidden, slide_tile_attn_weights = self.slide_aggregator(
                wsi_tokens,
                slide_ids,
                tile_attn_mask,
            )
            return wsi_case, None, slide_attn_weights, slide_hidden, slide_tile_attn_weights
        wsi_case, tile_pool_weights = self._pool_wsi_tokens(wsi_tokens, tile_attn_mask)
        return wsi_case, tile_pool_weights, None, None, None

    def _slide_level_b_points_feature(
        self,
        wsi_tokens: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        slide_ids: torch.Tensor | None,
        slide_attn_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.wsi_b_points_encoder is None or self.wsi_geo_feature_proj is None:
            raise RuntimeError("b_points requires wsi_b_points_encoder and wsi_geo_feature_proj")
        if slide_ids is None:
            raise RuntimeError("slide-level b_points requires slide_ids")
        batch_size = wsi_tokens.shape[0]
        valid_mask = tile_attn_mask.to(dtype=torch.bool)
        valid_slide_ids = slide_ids.masked_fill(~valid_mask, -1)
        max_slide_count = int(valid_slide_ids.max().item()) + 1 if bool(valid_mask.any()) else 1
        slide_features = wsi_tokens.new_zeros((batch_size, max_slide_count, int(self.hidden_dim)))
        slide_points = wsi_tokens.new_zeros(
            (batch_size, max_slide_count, self.effective_wsi_geo_num_points, self.effective_wsi_geo_coord_dim)
        )
        slide_valid = torch.zeros((batch_size, max_slide_count), dtype=torch.bool, device=wsi_tokens.device)
        for slide_idx in range(max_slide_count):
            slide_mask = (valid_slide_ids == slide_idx) & valid_mask
            if not bool(slide_mask.any()):
                continue
            slide_mask_f = slide_mask.to(dtype=tile_attn_mask.dtype)
            geometry = self.wsi_b_points_encoder(wsi_tokens, slide_mask_f, return_attn=False)
            slide_features[:, slide_idx, :] = self.wsi_geo_feature_proj(
                self._extract_curve_feature(geometry, geo_output=self.wsi_geo_output)
            )
            slide_points[:, slide_idx, :, :] = geometry["points"]
            slide_valid[:, slide_idx] = slide_mask.any(dim=1)
        if slide_attn_weights is not None and int(slide_attn_weights.shape[1]) == int(max_slide_count):
            slide_weights = slide_attn_weights
        else:
            slide_weights = slide_valid.to(dtype=torch.float32)
            denom = slide_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            slide_weights = slide_weights / denom
        feature = (slide_weights.unsqueeze(-1) * slide_features).sum(dim=1)
        aux = {
            "wsi_b_points_points_by_slide": slide_points,
            "wsi_b_points_slide_weights": slide_weights,
            "wsi_b_points_points": (slide_weights.unsqueeze(-1).unsqueeze(-1) * slide_points).sum(dim=1),
        }
        return feature, aux

    def _forward_custom(
        self,
        *,
        tile_tokens: torch.Tensor,
        tile_xy: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        slide_ids: torch.Tensor | None,
        rna_vec: torch.Tensor | None,
        rna_omics: list[torch.Tensor] | None,
        wsi_anti_vec: torch.Tensor | None = None,
        rna_anti_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wsi_tokens = self.wsi_proj(tile_tokens)
        rna_case, rna_aux = self._encode_rna(rna_vec=rna_vec, rna_omics=rna_omics)
        aux: dict[str, torch.Tensor] = {}
        aux.update(rna_aux)
        tile_pool_weights = None
        slide_attn_weights = None
        slide_hidden = None
        slide_tile_attn_weights = None
        wsi_parallel_feature = None
        rna_fused_feature = None

        if self.effective_wsi_geo_type != "none" and self.wsi_geo_position == "before_mean":
            if self.effective_wsi_geo_type == "curve":
                curve_source = _masked_mean(wsi_tokens, tile_attn_mask)
                wsi_geo_feature, wsi_geo_aux = self._custom_curve_feature(
                    self.wsi_curve_encoder,
                    self.wsi_geo_feature_proj,
                    curve_source,
                    geo_output=self.wsi_geo_output,
                )
            elif self.effective_wsi_geo_type == "b_points":
                wsi_geo_feature, wsi_geo_aux = self._custom_b_points_feature(
                    self.wsi_b_points_encoder,
                    self.wsi_geo_feature_proj,
                    wsi_tokens,
                    tile_attn_mask,
                    geo_output=self.wsi_geo_output,
                )
            else:
                wsi_geo_feature, wsi_geo_aux = self._custom_tile_feature(
                    self.wsi_geo_feature_proj,
                    tile_xy,
                    tile_attn_mask,
                    self.effective_wsi_geo_num_points,
                )
            aux.update({f"wsi_{k}": v for k, v in wsi_geo_aux.items()})
            if wsi_geo_feature is not None:
                geo_expand = wsi_geo_feature.unsqueeze(1).expand(-1, wsi_tokens.shape[1], -1)
                if self.wsi_geo_fusion == "concat":
                    if self.wsi_geo_before_align is None:
                        raise RuntimeError("before_mean concat requires alignment layer")
                    wsi_tokens = self.wsi_geo_before_align(torch.cat([wsi_tokens, geo_expand], dim=-1))
                else:
                    if self.wsi_geo_before_gate is None:
                        raise RuntimeError("before_mean gated requires gate layer")
                    gate = self.wsi_geo_before_gate(torch.cat([wsi_tokens, geo_expand], dim=-1))
                    wsi_tokens = gate * wsi_tokens + (1.0 - gate) * geo_expand
                    aux["wsi_geo_before_gate"] = gate

        wsi_case, tile_pool_weights, slide_attn_weights, slide_hidden, slide_tile_attn_weights = self._pool_multislide_tokens(
            wsi_tokens,
            tile_attn_mask,
            slide_ids,
        )

        wsi_case, w_anti_gate = self._apply_anti_injection(wsi_case, wsi_anti_vec, branch="wsi")
        if w_anti_gate is not None:
            aux["wsi_anti_gate"] = w_anti_gate

        if self.effective_wsi_geo_type != "none" and self.wsi_geo_position != "before_mean":
            if self.effective_wsi_geo_type == "curve":
                wsi_geo_feature, wsi_geo_aux = self._custom_curve_feature(
                    self.wsi_curve_encoder,
                    self.wsi_geo_feature_proj,
                    wsi_case,
                    geo_output=self.wsi_geo_output,
                )
            elif self.effective_wsi_geo_type == "b_points":
                if self.use_multi_slide and self.wsi_b_points_level == "slide":
                    wsi_geo_feature, wsi_geo_aux = self._slide_level_b_points_feature(
                        wsi_tokens,
                        tile_attn_mask,
                        slide_ids,
                        slide_attn_weights,
                    )
                else:
                    wsi_geo_feature, wsi_geo_aux = self._custom_b_points_feature(
                        self.wsi_b_points_encoder,
                        self.wsi_geo_feature_proj,
                        wsi_tokens,
                        tile_attn_mask,
                        geo_output=self.wsi_geo_output,
                    )
            else:
                wsi_geo_feature, wsi_geo_aux = self._custom_tile_feature(
                    self.wsi_geo_feature_proj,
                    tile_xy,
                    tile_attn_mask,
                    self.effective_wsi_geo_num_points,
                )
            aux.update({f"wsi_{k}": v for k, v in wsi_geo_aux.items()})
            if wsi_geo_feature is not None:
                if self.wsi_geo_position == "after_mean":
                    if self.wsi_geo_fusion == "concat":
                        if self.wsi_geo_after_align is None:
                            raise RuntimeError("after_mean concat requires alignment layer")
                        wsi_case = self.wsi_geo_after_align(torch.cat([wsi_case, wsi_geo_feature], dim=1))
                    else:
                        if self.wsi_geo_after_gate is None:
                            raise RuntimeError("after_mean gated requires gate layer")
                        gate = self.wsi_geo_after_gate(torch.cat([wsi_case, wsi_geo_feature], dim=1))
                        wsi_case = gate * wsi_case + (1.0 - gate) * wsi_geo_feature
                        aux["wsi_geo_after_gate"] = gate
                elif self.wsi_geo_position == "parallel":
                    if self.wsi_geo_fusion == "gated":
                        if self.wsi_geo_parallel_gate is None:
                            raise RuntimeError("parallel gated requires gate layer")
                        gate = self.wsi_geo_parallel_gate(torch.cat([wsi_case, wsi_geo_feature], dim=1))
                        wsi_parallel_feature = gate * wsi_case + (1.0 - gate) * wsi_geo_feature
                        aux["wsi_geo_parallel_gate"] = gate
                    else:
                        wsi_parallel_feature = wsi_geo_feature

        rna_case, r_anti_gate = self._apply_anti_injection(rna_case, rna_anti_vec, branch="rna")
        if r_anti_gate is not None:
            aux["rna_anti_gate"] = r_anti_gate

        gate_source = wsi_case
        if self.effective_rna_geo_type != "none":
            if self.effective_rna_geo_type == "curve":
                rna_geo_feature, rna_geo_aux = self._custom_curve_feature(
                    self.rna_curve_encoder,
                    self.rna_geo_feature_proj,
                    rna_case,
                    geo_output=self.effective_rna_geo_output,
                )
            else:
                rna_geo_feature, rna_geo_aux = self._custom_tile_feature(
                    self.rna_geo_feature_proj,
                    tile_xy,
                    tile_attn_mask,
                    self.effective_rna_geo_num_points,
                )
            aux.update({f"rna_{k}": v for k, v in rna_geo_aux.items()})
            if rna_geo_feature is not None:
                if self.rna_geo_position == "after_rna_proj":
                    rna_case = self._apply_rna_geo_fusion(rna_case, rna_geo_feature)
                elif self.rna_geo_position == "before_gate":
                    gate_source = self._apply_rna_geo_fusion(wsi_case, rna_geo_feature)
                elif self.rna_geo_position == "at_fused":
                    rna_fused_feature = self._apply_rna_geo_fusion(rna_case, rna_geo_feature)

        if self.gate_net is not None:
            gate = self.gate_net(gate_source)
            rna_case = rna_case * gate
        else:
            gate = torch.ones_like(rna_case)

        fused_main, fusion_aux = self._fuse_modalities(wsi_case=wsi_case, rna_case=rna_case)
        aux.update(fusion_aux)
        fused_parts = [fused_main]
        if wsi_parallel_feature is not None:
            fused_parts.append(wsi_parallel_feature)
        if rna_fused_feature is not None:
            fused_parts.append(rna_fused_feature)
        fused = torch.cat(fused_parts, dim=1)
        hidden = self.backbone(fused)
        risk = self.head(hidden).squeeze(-1)

        aux.update(
            {
                "wsi_case": wsi_case,
                "rna_case": rna_case,
                "fused": fused,
                "gate": gate,
            }
        )
        if wsi_parallel_feature is not None:
            aux["wsi_geo_parallel"] = wsi_parallel_feature
        if rna_fused_feature is not None:
            aux["rna_geo_fused"] = rna_fused_feature
        if tile_pool_weights is not None:
            aux["tile_pool_weights"] = tile_pool_weights
        if slide_attn_weights is not None:
            aux["slide_attn_weights"] = slide_attn_weights
        if slide_hidden is not None:
            aux["slide_hidden"] = slide_hidden
        if slide_tile_attn_weights is not None:
            aux["slide_tile_attn_weights"] = slide_tile_attn_weights
        return risk, aux

    def forward(
        self,
        *,
        tile_tokens: torch.Tensor,
        tile_xy: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        slide_ids: torch.Tensor | None,
        rna_vec: torch.Tensor | None,
        rna_omics: list[torch.Tensor] | None = None,
        wsi_anti_vec: torch.Tensor | None = None,
        rna_anti_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_custom:
            return self._forward_custom(
                tile_tokens=tile_tokens,
                tile_xy=tile_xy,
                tile_attn_mask=tile_attn_mask,
                slide_ids=slide_ids,
                rna_vec=rna_vec,
                rna_omics=rna_omics,
                wsi_anti_vec=wsi_anti_vec,
                rna_anti_vec=rna_anti_vec,
            )

        wsi_tokens = self.wsi_proj(tile_tokens)
        b_aux: dict[str, torch.Tensor] = {}
        tile_pool_weights = None
        slide_attn_weights = None
        slide_hidden = None
        slide_tile_attn_weights = None
        geo_case = None
        geo_mix_gate = None

        if self.effective_wsi_geo_type == "b_points" and self.wsi_geo_position == "before_mean":
            wsi_geo_feature, wsi_geo_aux = self._custom_b_points_feature(
                self.wsi_b_points_encoder,
                self.wsi_geo_feature_proj,
                wsi_tokens,
                tile_attn_mask,
                geo_output=self.wsi_geo_output,
            )
            b_aux.update({f"wsi_{k}": v for k, v in wsi_geo_aux.items()})
            if wsi_geo_feature is not None:
                geo_expand = wsi_geo_feature.unsqueeze(1).expand(-1, wsi_tokens.shape[1], -1)
                if self.wsi_geo_fusion == "concat":
                    if self.wsi_geo_before_align is None:
                        raise RuntimeError("before_mean concat requires alignment layer")
                    wsi_tokens = self.wsi_geo_before_align(torch.cat([wsi_tokens, geo_expand], dim=-1))
                else:
                    if self.wsi_geo_before_gate is None:
                        raise RuntimeError("before_mean gated requires gate layer")
                    gate = self.wsi_geo_before_gate(torch.cat([wsi_tokens, geo_expand], dim=-1))
                    wsi_tokens = gate * wsi_tokens + (1.0 - gate) * geo_expand
                    b_aux["wsi_geo_before_gate"] = gate

        if self.use_geo_tile:
            if self.geo_proj is None or self.tile_fusion_proj is None:
                raise RuntimeError("geo_tile requires geo_proj and tile_fusion_proj")
            geo_tokens = _build_geo_tokens(tile_xy, tile_attn_mask)
            geo_hidden = self.geo_proj(geo_tokens)
            geo_case = _masked_mean(geo_hidden, tile_attn_mask)
            tile_fused = torch.cat([wsi_tokens, geo_hidden], dim=-1)
            wsi_tokens = self.tile_fusion_proj(tile_fused)
            if self.use_multi_slide:
                if slide_ids is None or self.slide_aggregator is None:
                    raise RuntimeError("multi-slide mode requires slide_ids and slide_aggregator")
                wsi_case, slide_attn_weights, slide_hidden, slide_tile_attn_weights = self.slide_aggregator(wsi_tokens, slide_ids, tile_attn_mask)
            else:
                wsi_case, tile_pool_weights = self._pool_wsi_tokens(wsi_tokens, tile_attn_mask)
        else:
            if self.use_multi_slide:
                if slide_ids is None or self.slide_aggregator is None:
                    raise RuntimeError("multi-slide mode requires slide_ids and slide_aggregator")
                wsi_case, slide_attn_weights, slide_hidden, slide_tile_attn_weights = self.slide_aggregator(wsi_tokens, slide_ids, tile_attn_mask)
            else:
                wsi_case, tile_pool_weights = self._pool_wsi_tokens(wsi_tokens, tile_attn_mask)

        if self.effective_wsi_geo_type == "b_points" and self.wsi_geo_position == "after_mean":
            wsi_geo_feature = None
            if self.wsi_b_points_level == "slide":
                wsi_geo_feature, wsi_geo_aux = self._slide_level_b_points_feature(
                    wsi_tokens,
                    tile_attn_mask,
                    slide_ids,
                    slide_attn_weights,
                )
                b_aux.update(wsi_geo_aux)
            else:
                wsi_geo_feature, wsi_geo_aux = self._custom_b_points_feature(
                    self.wsi_b_points_encoder,
                    self.wsi_geo_feature_proj,
                    wsi_tokens,
                    tile_attn_mask,
                    geo_output=self.wsi_geo_output,
                )
                b_aux.update({f"wsi_{k}": v for k, v in wsi_geo_aux.items()})
            if wsi_geo_feature is not None:
                if self.wsi_geo_fusion == "concat":
                    if self.wsi_geo_after_align is None:
                        raise RuntimeError("after_mean concat requires alignment layer")
                    wsi_case = self.wsi_geo_after_align(torch.cat([wsi_case, wsi_geo_feature], dim=1))
                else:
                    if self.wsi_geo_after_gate is None:
                        raise RuntimeError("after_mean gated requires gate layer")
                    gate = self.wsi_geo_after_gate(torch.cat([wsi_case, wsi_geo_feature], dim=1))
                    wsi_case = gate * wsi_case + (1.0 - gate) * wsi_geo_feature
                    b_aux["wsi_geo_after_gate"] = gate

        wsi_case, w_gate = self._apply_anti_injection(wsi_case, wsi_anti_vec, branch="wsi")
        if w_gate is not None:
            b_aux["wsi_anti_gate"] = w_gate

        rna_case, rna_aux = self._encode_rna(rna_vec=rna_vec, rna_omics=rna_omics)
        b_aux.update(rna_aux)

        rna_case, r_gate = self._apply_anti_injection(rna_case, rna_anti_vec, branch="rna")
        if r_gate is not None:
            b_aux["rna_anti_gate"] = r_gate

        if self.gate_net is not None:
            gate = self.gate_net(wsi_case)
            rna_case = rna_case * gate
        else:
            gate = torch.ones_like(rna_case)

        aug_feat = None
        if self.aug_proj is not None and self.aug_align is not None:
            token_max = _masked_max(wsi_tokens, tile_attn_mask)
            aug_feat = self.aug_proj(torch.cat([wsi_case, token_max], dim=1))
            wsi_case = self.aug_align(torch.cat([wsi_case, aug_feat], dim=1))

        if self.geo_proj is not None and self.geo_align is not None and not self.use_geo_tile:
            geo_tokens = _build_geo_tokens(tile_xy, tile_attn_mask)
            geo_hidden = self.geo_proj(geo_tokens)
            geo_case = _masked_mean(geo_hidden, tile_attn_mask)
            if self.use_geo_gate_v2 and self.geo_mix_gate is not None:
                geo_case_aligned = self.geo_align(geo_case)
                geo_mix_gate = self.geo_mix_gate(torch.cat([wsi_case, geo_case_aligned], dim=1))
                wsi_case = geo_mix_gate * wsi_case + (1.0 - geo_mix_gate) * geo_case_aligned
                geo_case = geo_case_aligned
            else:
                wsi_case = self.geo_align(torch.cat([wsi_case, geo_case], dim=1))

        fused, fusion_aux = self._fuse_modalities(wsi_case=wsi_case, rna_case=rna_case)
        hidden = self.backbone(fused)
        risk = self.head(hidden).squeeze(-1)
        aux = {
            "wsi_case": wsi_case,
            "rna_case": rna_case,
            "fused": fused,
            "gate": gate,
        }
        aux.update(fusion_aux)
        aux.update(b_aux)
        if aug_feat is not None:
            aux["wsi_aug"] = aug_feat
        if geo_case is not None:
            aux["wsi_geo"] = geo_case
        if geo_mix_gate is not None:
            aux["geo_mix_gate"] = geo_mix_gate
        if tile_pool_weights is not None:
            aux["tile_pool_weights"] = tile_pool_weights
        if slide_attn_weights is not None:
            aux["slide_attn_weights"] = slide_attn_weights
        if slide_hidden is not None:
            aux["slide_hidden"] = slide_hidden
        if slide_tile_attn_weights is not None:
            aux["slide_tile_attn_weights"] = slide_tile_attn_weights
        return risk, aux
