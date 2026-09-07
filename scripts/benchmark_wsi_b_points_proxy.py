from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data.build_index import IndexRow
from r2wsp.data.build_index import build_index
from r2wsp.data.paths import resolve_data_paths
from r2wsp.data.scan_assets import scan_all_assets
from r2wsp.data.wsi_input import load_wsi_input
from r2wsp.models.direct_survival import TokenPointGeometryEncoder, geometry_per_point_tensor, local_geometry


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_official_split(split_dir: Path, target_col: str) -> tuple[dict[str, dict[str, float]], set[str], set[str]]:
    censorship_col = target_col.split("_")[0] + "_censorship"
    train_rows = read_split_csv(split_dir / "train.csv")
    test_rows = read_split_csv(split_dir / "test.csv")
    case_table: dict[str, dict[str, float]] = {}
    for row in train_rows + test_rows:
        case_id = str(row["case_id"])
        case_table[case_id] = {
            "event_time": float(row[target_col]),
            "censorship": float(row[censorship_col]),
        }
    return case_table, {str(r["case_id"]) for r in train_rows}, {str(r["case_id"]) for r in test_rows}


def _build_survival_strata(
    train_case_ids: list[str],
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int,
) -> dict[str, list[str]]:
    case_ids = [str(c) for c in train_case_ids]
    event_times = np.asarray([float(case_table[str(c)]["event_time"]) for c in train_case_ids], dtype=np.float64)
    censorships = np.asarray([float(case_table[str(c)]["censorship"]) for c in train_case_ids], dtype=np.float64)
    uncensored = event_times[censorships < 0.5]
    if uncensored.size >= 2 and int(n_time_bins) > 1:
        quantiles = np.quantile(uncensored, np.linspace(0.0, 1.0, int(n_time_bins) + 1))
        quantiles[0] = min(float(quantiles[0]), float(event_times.min())) - 1e-6
        quantiles[-1] = max(float(quantiles[-1]), float(event_times.max())) + 1e-6
        quantiles = np.unique(quantiles)
        if quantiles.size >= 3:
            time_bins = np.digitize(event_times, quantiles[1:-1], right=False).astype(int)
        else:
            time_bins = np.zeros(len(case_ids), dtype=int)
    else:
        time_bins = np.zeros(len(case_ids), dtype=int)
    strata: dict[str, list[str]] = {}
    for case_id, censorship, time_bin in zip(case_ids, censorships, time_bins):
        stratum = f"{int(censorship)}_{int(time_bin)}"
        strata.setdefault(stratum, []).append(case_id)
    return strata


def split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int = 4,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    strata = _build_survival_strata(train_case_ids, case_table, n_time_bins=int(n_time_bins))
    val_ids: list[str] = []
    train_ids: list[str] = []
    for stratum in sorted(strata.keys()):
        ids = np.asarray(sorted(strata[stratum]), dtype=object)
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.extend(str(x) for x in ids.tolist())
            continue
        val_size = int(round(len(ids) * float(val_frac)))
        val_size = max(1, min(len(ids) - 1, val_size))
        val_ids.extend(str(x) for x in ids[:val_size].tolist())
        train_ids.extend(str(x) for x in ids[val_size:].tolist())
    return set(train_ids), set(val_ids)


def split_fixed_stratified_case_ids(
    case_ids: list[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    case_table: dict[str, dict[str, float]],
    n_time_bins: int,
) -> tuple[set[str], set[str], set[str]]:
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if not (abs(total - 1.0) < 1e-6):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    rng = np.random.default_rng(int(seed))
    strata = _build_survival_strata(case_ids, case_table, n_time_bins=int(n_time_bins))
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for stratum in sorted(strata.keys()):
        ids = np.asarray(sorted(strata[stratum]), dtype=object)
        rng.shuffle(ids)
        n = int(len(ids))
        if n <= 2:
            train_ids.extend(str(x) for x in ids.tolist())
            continue
        n_val = int(round(n * float(val_frac)))
        n_test = int(round(n * float(test_frac)))
        n_val = max(1, min(n - 2, n_val))
        n_test = max(1, min(n - 1 - n_val, n_test))
        val_ids.extend(str(x) for x in ids[:n_val].tolist())
        test_ids.extend(str(x) for x in ids[n_val : n_val + n_test].tolist())
        train_ids.extend(str(x) for x in ids[n_val + n_test :].tolist())
    return set(train_ids), set(val_ids), set(test_ids)


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


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(dtype=torch.float32).unsqueeze(-1)
    s = (x * w).sum(dim=1)
    d = w.sum(dim=1).clamp_min(1.0)
    return s / d


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


@dataclass(frozen=True)
class TokenSample:
    case_id: str
    tile_tokens: torch.Tensor
    tile_attn_mask: torch.Tensor
    slide_ids: torch.Tensor


class WSITokenCaseDataset(Dataset):
    def __init__(
        self,
        *,
        groups: dict[str, list[IndexRow]],
        case_ids: list[str],
        max_tiles: int,
        use_multi_slide: bool,
        multi_slide_tile_budget_mode: str,
    ) -> None:
        self.groups = groups
        self.case_ids = case_ids
        self.max_tiles = int(max_tiles)
        self.use_multi_slide = bool(use_multi_slide)
        self.multi_slide_tile_budget_mode = str(multi_slide_tile_budget_mode)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> TokenSample:
        case_id = str(self.case_ids[idx])
        rows = self.groups[case_id]
        sorted_rows = sorted(rows, key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))
        if self.use_multi_slide and self.multi_slide_tile_budget_mode == "case_shared" and self.max_tiles > 0:
            n_slides = max(len(sorted_rows), 1)
            base = max(1, int(self.max_tiles) // n_slides)
            remainder = max(0, int(self.max_tiles) - base * n_slides)
            tile_budgets = [base + (1 if i < remainder else 0) for i in range(n_slides)]
        else:
            tile_budgets = [int(self.max_tiles) if int(self.max_tiles) > 0 else -1 for _ in range(len(sorted_rows))]
        token_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        slide_ids_list: list[torch.Tensor] = []
        for slide_idx, r in enumerate(sorted_rows):
            max_tiles_this_slide = tile_budgets[slide_idx]
            tile_tokens, _, tile_attn_mask = load_wsi_input(
                Path(r.wsi_feature_path),
                kind=str(r.wsi_feature_kind),
                source_name=str(r.wsi_feature_source),
                max_tiles=max_tiles_this_slide if max_tiles_this_slide > 0 else None,
            )
            n = int(tile_tokens.shape[0])
            token_list.append(tile_tokens)
            mask_list.append(tile_attn_mask)
            slide_ids_list.append(torch.full((n,), fill_value=int(slide_idx), dtype=torch.long))
            if not self.use_multi_slide:
                break
        tile_tokens = torch.cat(token_list, dim=0)
        tile_attn_mask = torch.cat(mask_list, dim=0)
        slide_ids = torch.cat(slide_ids_list, dim=0)
        return TokenSample(case_id=case_id, tile_tokens=tile_tokens, tile_attn_mask=tile_attn_mask, slide_ids=slide_ids)


def collate_token_samples(samples: list[TokenSample]) -> dict[str, object]:
    case_ids = [s.case_id for s in samples]
    max_n = max(int(s.tile_tokens.shape[0]) for s in samples)
    d = int(samples[0].tile_tokens.shape[1])
    tile_tokens = torch.zeros((len(samples), max_n, d), dtype=torch.float32)
    tile_attn_mask = torch.zeros((len(samples), max_n), dtype=torch.long)
    slide_ids = torch.full((len(samples), max_n), fill_value=-1, dtype=torch.long)
    for i, s in enumerate(samples):
        n = int(s.tile_tokens.shape[0])
        tile_tokens[i, :n] = s.tile_tokens
        tile_attn_mask[i, :n] = s.tile_attn_mask
        slide_ids[i, :n] = s.slide_ids
    return {
        "case_id": case_ids,
        "tile_tokens": tile_tokens,
        "tile_attn_mask": tile_attn_mask,
        "slide_ids": slide_ids,
    }


class TinyWSIBPointsCox(nn.Module):
    def __init__(
        self,
        *,
        tile_dim: int,
        hidden_dim: int,
        dropout: float,
        method: str,
        num_points: int,
        b_points_level: str,
        point_feature_dim: int,
        fusion_mode: str,
    ) -> None:
        super().__init__()
        if method not in {"mean", "b_points", "var_interleave", "free_points", "weighted_points"}:
            raise ValueError("method must be mean|b_points|var_interleave|free_points|weighted_points")
        if b_points_level not in {"case", "slide"}:
            raise ValueError("b_points_level must be case|slide")
        if fusion_mode not in {
            "concat",
            "gated",
            "geo_residual",
            "tile_self_attn",
            "tile_cross_attn",
            "refine_cross_attn",
            "slide_case_attn",
            "guided_slide_case_attn",
        }:
            raise ValueError(
                "fusion_mode must be concat|gated|geo_residual|tile_self_attn|tile_cross_attn|refine_cross_attn|slide_case_attn|guided_slide_case_attn"
            )
        self.method = str(method)
        self.num_points = int(num_points)
        self.b_points_level = str(b_points_level)
        self.hidden_dim = int(hidden_dim)
        self.point_feature_dim = int(point_feature_dim)
        self.fusion_mode = str(fusion_mode)
        self.branch1_dim = (int(hidden_dim) + 1) // 2
        self.branch2_dim = int(hidden_dim) // 2
        self.wsi_proj = nn.Sequential(nn.Linear(int(tile_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
        self.var_branch1_proj = nn.Sequential(nn.Linear(int(self.branch1_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
        self.var_branch2_proj = nn.Sequential(nn.Linear(int(self.branch2_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
        self.free_point_head = nn.Linear(int(hidden_dim), int(num_points) * 3)
        self.weighted_point_head = nn.Linear(int(hidden_dim), int(num_points) * 4)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, int(self.point_feature_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.point_hidden_proj = nn.Sequential(
            nn.Linear(int(self.point_feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.points_encoder = TokenPointGeometryEncoder(int(hidden_dim), int(num_points))
        self.slide_attn_score = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), 1))
        if self.fusion_mode in {"concat", "slide_case_attn"}:
            self.fuse = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
            self.geo_gate = None
            self.geo_tile_gate = None
            self.geo_tile_align = None
            self.fused_align = None
            self.visual_refine_attn = None
            self.visual_cross_attn = None
            self.visual_attn_ln = None
        elif self.fusion_mode == "guided_slide_case_attn":
            self.fuse = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
            self.geo_gate = None
            self.geo_tile_gate = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
                nn.Sigmoid(),
            )
            self.geo_tile_align = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
            self.fused_align = None
            self.visual_refine_attn = None
            self.visual_cross_attn = None
            self.visual_attn_ln = nn.LayerNorm(int(hidden_dim))
        else:
            self.fuse = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
            if self.fusion_mode == "gated":
                self.geo_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.Sigmoid())
                self.geo_tile_gate = None
                self.geo_tile_align = None
                self.fused_align = None
                self.visual_refine_attn = None
                self.visual_cross_attn = None
                self.visual_attn_ln = None
            elif self.fusion_mode == "geo_residual":
                self.geo_gate = nn.Sequential(nn.Linear(int(hidden_dim) * 2, 1), nn.Sigmoid())
                self.geo_tile_gate = None
                self.geo_tile_align = None
                self.fused_align = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout)))
                self.visual_refine_attn = None
                self.visual_cross_attn = None
                self.visual_attn_ln = None
            else:
                self.geo_gate = None
                self.geo_tile_gate = None
                self.geo_tile_align = None
                self.fused_align = None
                self.visual_refine_attn = nn.MultiheadAttention(int(hidden_dim), num_heads=4, dropout=float(dropout), batch_first=True)
                self.visual_cross_attn = nn.MultiheadAttention(int(hidden_dim), num_heads=4, dropout=float(dropout), batch_first=True)
                self.visual_attn_ln = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), 1)

    def _hierarchical_visual_only(
        self, wsi_tokens: torch.Tensor, tile_attn_mask: torch.Tensor, slide_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = wsi_tokens.shape[0]
        valid_mask = tile_attn_mask.to(dtype=torch.bool)
        valid_slide_ids = slide_ids.masked_fill(~valid_mask, -1)
        max_slide_count = int(valid_slide_ids.max().item()) + 1 if bool(valid_mask.any()) else 1
        slide_repr = wsi_tokens.new_zeros((batch_size, max_slide_count, self.hidden_dim))
        slide_valid = torch.zeros((batch_size, max_slide_count), dtype=torch.bool, device=wsi_tokens.device)
        for slide_idx in range(max_slide_count):
            slide_mask = (valid_slide_ids == slide_idx) & valid_mask
            if not bool(slide_mask.any()):
                continue
            slide_mask_f = slide_mask.to(dtype=wsi_tokens.dtype)
            slide_case = masked_mean(wsi_tokens, slide_mask_f)
            slide_repr[:, slide_idx, :] = slide_case
            slide_valid[:, slide_idx] = slide_mask.any(dim=1)
        attn_logits = self.slide_attn_score(slide_repr).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~slide_valid, float("-inf"))
        all_invalid = ~slide_valid.any(dim=1, keepdim=True)
        attn_logits = torch.where(all_invalid, torch.zeros_like(attn_logits), attn_logits)
        slide_weights = torch.softmax(attn_logits, dim=1)
        slide_weights = slide_weights * slide_valid.to(dtype=slide_weights.dtype)
        slide_weights = slide_weights / slide_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        case_repr = (slide_weights.unsqueeze(-1) * slide_repr).sum(dim=1)
        aux: dict[str, torch.Tensor] = {"slide_case_weights": slide_weights}
        risk = self.head(case_repr).squeeze(-1)
        return risk, aux

    def _hierarchical_points(
        self, wsi_tokens: torch.Tensor, tile_attn_mask: torch.Tensor, slide_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = wsi_tokens.shape[0]
        valid_mask = tile_attn_mask.to(dtype=torch.bool)
        valid_slide_ids = slide_ids.masked_fill(~valid_mask, -1)
        max_slide_count = int(valid_slide_ids.max().item()) + 1 if bool(valid_mask.any()) else 1
        slide_repr = wsi_tokens.new_zeros((batch_size, max_slide_count, self.hidden_dim))
        slide_valid = torch.zeros((batch_size, max_slide_count), dtype=torch.bool, device=wsi_tokens.device)
        slide_points = wsi_tokens.new_zeros((batch_size, max_slide_count, self.num_points, 3))
        slide_point_weights = wsi_tokens.new_full((batch_size, max_slide_count, self.num_points), 1.0 / max(self.num_points, 1))
        smooth_l1_terms: list[torch.Tensor] = []
        smooth_l2_terms: list[torch.Tensor] = []
        gate_mean_terms: list[torch.Tensor] = []
        for slide_idx in range(max_slide_count):
            slide_mask = (valid_slide_ids == slide_idx) & valid_mask
            if not bool(slide_mask.any()):
                continue
            slide_mask_f = slide_mask.to(dtype=wsi_tokens.dtype)
            slide_case = masked_mean(wsi_tokens, slide_mask_f)
            if self.method == "free_points":
                points = self.free_point_head(slide_case).view(batch_size, self.num_points, 3)
                point_features = self.point_hidden_proj(self.point_encoder(points))
                geo_case = point_features.mean(dim=1)
            elif self.method == "weighted_points":
                weighted = self.weighted_point_head(slide_case).view(batch_size, self.num_points, 4)
                points = weighted[:, :, :3]
                point_weights = torch.softmax(weighted[:, :, 3], dim=1)
                point_features = self.point_hidden_proj(self.point_encoder(points))
                geo_case = (point_weights.unsqueeze(-1) * point_features).sum(dim=1)
                slide_point_weights[:, slide_idx, :] = point_weights
            else:
                raise RuntimeError("slide_case_attn only supports free_points|weighted_points")
            if self.fusion_mode == "guided_slide_case_attn":
                if self.geo_tile_gate is None or self.geo_tile_align is None or self.visual_attn_ln is None:
                    raise RuntimeError("guided_slide_case_attn requires tile guidance layers")
                geo_context = self.geo_tile_align(geo_case).unsqueeze(1).expand(-1, wsi_tokens.shape[1], -1)
                gate_input = torch.cat([wsi_tokens, geo_context], dim=-1)
                tile_gate = self.geo_tile_gate(gate_input).squeeze(-1)
                tile_gate = tile_gate * slide_mask.to(dtype=tile_gate.dtype)
                gate_mean_terms.append(tile_gate.sum(dim=1) / slide_mask_f.sum(dim=1).clamp_min(1.0))
                guided_tokens = wsi_tokens + tile_gate.unsqueeze(-1) * geo_context
                guided_tokens = self.visual_attn_ln(guided_tokens)
                fused_slide = self.fuse(masked_mean(guided_tokens, slide_mask.to(dtype=wsi_tokens.dtype)))
            else:
                fused_slide = self.fuse(torch.cat([slide_case, geo_case], dim=1))
            slide_repr[:, slide_idx, :] = fused_slide
            slide_valid[:, slide_idx] = slide_mask.any(dim=1)
            slide_points[:, slide_idx, :, :] = points
            d1 = points[:, 1:, :] - points[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            smooth_l1_terms.append((d1 ** 2).mean(dim=(1, 2)))
            smooth_l2_terms.append((d2 ** 2).mean(dim=(1, 2)))

        attn_logits = self.slide_attn_score(slide_repr).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~slide_valid, float("-inf"))
        all_invalid = ~slide_valid.any(dim=1, keepdim=True)
        attn_logits = torch.where(all_invalid, torch.zeros_like(attn_logits), attn_logits)
        slide_weights = torch.softmax(attn_logits, dim=1)
        slide_weights = slide_weights * slide_valid.to(dtype=slide_weights.dtype)
        slide_weights = slide_weights / slide_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        case_repr = (slide_weights.unsqueeze(-1) * slide_repr).sum(dim=1)
        points = (slide_weights.unsqueeze(-1).unsqueeze(-1) * slide_points).sum(dim=1)
        point_weights = (slide_weights.unsqueeze(-1) * slide_point_weights).sum(dim=1)
        point_weights = point_weights / point_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        regs = geometry_regularizers(points)
        curvature, torsion = local_geometry(points)
        curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
        aux: dict[str, torch.Tensor] = {
            "points": points,
            "curvature": curvature,
            "torsion": torsion,
            "geometry_per_point": geometry_per_point,
            "curvature_per_point": curvature_full,
            "torsion_per_point": torsion_full,
            "slide_case_weights": slide_weights,
        }
        if gate_mean_terms:
            aux["tile_guidance_gate_mean"] = (slide_weights * torch.stack(gate_mean_terms, dim=1)).sum(dim=1)
        aux["curvature_l2"] = regs["curvature_l2"]
        aux["torsion_l2"] = regs["torsion_l2"]
        if self.method == "weighted_points":
            aux["point_weights"] = point_weights
        if smooth_l1_terms:
            smooth_l1 = torch.stack(smooth_l1_terms, dim=1)
            smooth_l2 = torch.stack(smooth_l2_terms, dim=1)
            aux["smooth_l1"] = (slide_weights * smooth_l1).sum(dim=1).mean()
            aux["smooth_l2"] = (slide_weights * smooth_l2).sum(dim=1).mean()
        else:
            aux["smooth_l1"] = wsi_tokens.new_tensor(0.0)
            aux["smooth_l2"] = wsi_tokens.new_tensor(0.0)
        risk = self.head(case_repr).squeeze(-1)
        return risk, aux

    def forward(self, tile_tokens: torch.Tensor, tile_attn_mask: torch.Tensor, slide_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wsi_tokens = self.wsi_proj(tile_tokens)
        wsi_case = masked_mean(wsi_tokens, tile_attn_mask)
        aux: dict[str, torch.Tensor] = {}
        if self.method == "mean":
            if self.fusion_mode == "slide_case_attn":
                return self._hierarchical_visual_only(wsi_tokens, tile_attn_mask, slide_ids)
            risk = self.head(wsi_case).squeeze(-1)
            return risk, aux
        if self.method in {"free_points", "weighted_points"} and self.fusion_mode in {"slide_case_attn", "guided_slide_case_attn"}:
            return self._hierarchical_points(wsi_tokens, tile_attn_mask, slide_ids)
        if self.method == "var_interleave":
            if self.fusion_mode not in {"concat", "gated", "geo_residual"}:
                raise RuntimeError("var_interleave only supports concat|gated|geo_residual")
            mask = tile_attn_mask.to(dtype=wsi_tokens.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            token_mean = (wsi_tokens * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            token_var = (((wsi_tokens - token_mean) ** 2) * mask).sum(dim=1) / denom
            sorted_idx = torch.argsort(token_var, dim=1, descending=True)
            idx1 = sorted_idx[:, 0::2]
            idx2 = sorted_idx[:, 1::2]
            branch1 = torch.gather(wsi_case, 1, idx1)
            branch2 = torch.gather(wsi_case, 1, idx2)
            wsi_case = self.var_branch1_proj(branch1)
            geo_case = self.var_branch2_proj(branch2)
            aux["var_branch1_indices"] = idx1
            aux["var_branch2_indices"] = idx2
            point_tokens = None
        else:
            point_tokens = None

        if self.method == "free_points":
            points = self.free_point_head(wsi_case).view(wsi_case.shape[0], self.num_points, 3)
            geo_case = self.point_hidden_proj(self.point_encoder(points).mean(dim=1))
            aux["points"] = points
            regs = geometry_regularizers(points)
            aux["smooth_l1"] = regs["smooth_l1"]
            aux["smooth_l2"] = regs["smooth_l2"]
            aux["curvature_l2"] = regs["curvature_l2"]
            aux["torsion_l2"] = regs["torsion_l2"]
            curvature, torsion = local_geometry(points)
            curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
            aux["curvature"] = curvature
            aux["torsion"] = torsion
            aux["geometry_per_point"] = geometry_per_point
            aux["curvature_per_point"] = curvature_full
            aux["torsion_per_point"] = torsion_full
            point_tokens = self.point_hidden_proj(self.point_encoder(points))
        elif self.method == "weighted_points":
            weighted = self.weighted_point_head(wsi_case).view(wsi_case.shape[0], self.num_points, 4)
            points = weighted[:, :, :3]
            point_weights = torch.softmax(weighted[:, :, 3], dim=1)
            point_features = self.point_hidden_proj(self.point_encoder(points))
            geo_case = (point_weights.unsqueeze(-1) * point_features).sum(dim=1)
            aux["points"] = points
            aux["point_weights"] = point_weights
            regs = geometry_regularizers(points)
            aux["smooth_l1"] = regs["smooth_l1"]
            aux["smooth_l2"] = regs["smooth_l2"]
            aux["curvature_l2"] = regs["curvature_l2"]
            aux["torsion_l2"] = regs["torsion_l2"]
            curvature, torsion = local_geometry(points)
            curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
            aux["curvature"] = curvature
            aux["torsion"] = torsion
            aux["geometry_per_point"] = geometry_per_point
            aux["curvature_per_point"] = curvature_full
            aux["torsion_per_point"] = torsion_full
            point_tokens = point_features * point_weights.unsqueeze(-1)
        elif self.method == "b_points" and self.b_points_level == "slide":
            batch_size = wsi_tokens.shape[0]
            valid_mask = tile_attn_mask.to(dtype=torch.bool)
            valid_slide_ids = slide_ids.masked_fill(~valid_mask, -1)
            max_slide_count = int(valid_slide_ids.max().item()) + 1 if bool(valid_mask.any()) else 1
            slide_points = wsi_tokens.new_zeros((batch_size, max_slide_count, self.num_points, 3))
            slide_geo_case = wsi_tokens.new_zeros((batch_size, max_slide_count, self.hidden_dim))
            slide_valid = torch.zeros((batch_size, max_slide_count), dtype=torch.bool, device=wsi_tokens.device)
            for slide_idx in range(max_slide_count):
                slide_mask = (valid_slide_ids == slide_idx) & valid_mask
                if not bool(slide_mask.any()):
                    continue
                slide_mask_f = slide_mask.to(dtype=tile_attn_mask.dtype)
                geo = self.points_encoder(wsi_tokens, slide_mask_f, return_attn=False)
                points = geo["points"]
                slide_points[:, slide_idx, :, :] = points
                slide_geo_case[:, slide_idx, :] = self.point_hidden_proj(self.point_encoder(points).mean(dim=1))
                slide_valid[:, slide_idx] = slide_mask.any(dim=1)
            slide_weights = slide_valid.to(dtype=torch.float32)
            slide_weights = slide_weights / slide_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            geo_case = (slide_weights.unsqueeze(-1) * slide_geo_case).sum(dim=1)
            points = (slide_weights.unsqueeze(-1).unsqueeze(-1) * slide_points).sum(dim=1)
            aux["points"] = points
            curvature, torsion = local_geometry(points)
            curvature_full, torsion_full, geometry_per_point = geometry_per_point_tensor(points)
            aux["curvature"] = curvature
            aux["torsion"] = torsion
            aux["geometry_per_point"] = geometry_per_point
            aux["curvature_per_point"] = curvature_full
            aux["torsion_per_point"] = torsion_full
        else:
            geo = self.points_encoder(wsi_tokens, tile_attn_mask, return_attn=False)
            points = geo["points"]
            geo_case = self.point_hidden_proj(self.point_encoder(points).mean(dim=1))
            aux["points"] = points
            aux["curvature"] = geo["curvature"]
            aux["torsion"] = geo["torsion"]
            aux["geometry_per_point"] = geo["geometry_per_point"]
            aux["curvature_per_point"] = geo["curvature_per_point"]
            aux["torsion_per_point"] = geo["torsion_per_point"]
            point_tokens = self.point_hidden_proj(self.point_encoder(points))

        if self.fusion_mode == "concat":
            fused = self.fuse(torch.cat([wsi_case, geo_case], dim=1))
        elif self.fusion_mode == "gated":
            if self.geo_gate is None:
                raise RuntimeError("gated fusion requires geo_gate")
            gate = self.geo_gate(torch.cat([wsi_case, geo_case], dim=1))
            fused = self.fuse(gate * wsi_case + (1.0 - gate) * geo_case)
            aux["geo_fusion_gate"] = gate
        elif self.fusion_mode == "geo_residual":
            if self.geo_gate is None or self.fused_align is None:
                raise RuntimeError("geo_residual fusion requires geo_gate and fused_align")
            gate = self.geo_gate(torch.cat([wsi_case, geo_case], dim=1))
            fused = self.fuse(self.fused_align(wsi_case + gate * geo_case))
            aux["geo_residual_gate"] = gate
        elif self.fusion_mode == "tile_self_attn":
            if self.visual_refine_attn is None or self.visual_attn_ln is None:
                raise RuntimeError("tile_self_attn requires attention layers")
            visual_valid = tile_attn_mask.to(dtype=torch.bool)
            point_valid = torch.ones((point_tokens.shape[0], point_tokens.shape[1]), dtype=torch.bool, device=point_tokens.device)
            seq = torch.cat([wsi_tokens, point_tokens], dim=1)
            pad_mask = ~torch.cat([visual_valid, point_valid], dim=1)
            seq_out, _ = self.visual_refine_attn(seq, seq, seq, key_padding_mask=pad_mask, need_weights=False)
            seq = self.visual_attn_ln(seq + seq_out)
            fused = self.fuse(masked_mean(seq[:, : wsi_tokens.shape[1], :], tile_attn_mask))
        else:
            if self.visual_refine_attn is None or self.visual_cross_attn is None or self.visual_attn_ln is None:
                raise RuntimeError("attention fusion requires attention layers")
            visual_valid = tile_attn_mask.to(dtype=torch.bool)
            visual_tokens = wsi_tokens
            if self.fusion_mode == "refine_cross_attn":
                self_out, _ = self.visual_refine_attn(
                    visual_tokens,
                    visual_tokens,
                    visual_tokens,
                    key_padding_mask=~visual_valid,
                    need_weights=False,
                )
                visual_tokens = self.visual_attn_ln(visual_tokens + self_out)
            cross_out, _ = self.visual_cross_attn(visual_tokens, point_tokens, point_tokens, need_weights=False)
            visual_tokens = self.visual_attn_ln(visual_tokens + cross_out)
            fused = self.fuse(masked_mean(visual_tokens, tile_attn_mask))
        risk = self.head(fused).squeeze(-1)
        return risk, aux


def evaluate(
    model: TinyWSIBPointsCox,
    loader: DataLoader,
    case_table: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    risks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    censorships: list[np.ndarray] = []
    geom_case_curv: list[np.ndarray] = []
    geom_case_tors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            case_ids = batch["case_id"]
            tt = batch["tile_tokens"].to(device)
            am = batch["tile_attn_mask"].to(device)
            sid = batch["slide_ids"].to(device)
            t = torch.tensor([case_table[str(c)]["event_time"] for c in case_ids], device=device, dtype=torch.float32)
            c = torch.tensor([case_table[str(c)]["censorship"] for c in case_ids], device=device, dtype=torch.float32)
            e = 1.0 - c
            risk, aux = model(tt, am, sid)
            loss = neg_partial_log_likelihood(risk, t, e)
            losses.append(float(loss.detach().cpu()))
            risks.append(risk.detach().cpu().numpy())
            times.append(t.detach().cpu().numpy())
            censorships.append(c.detach().cpu().numpy())
            if "curvature" in aux:
                case_curv = aux["curvature"].mean(dim=1).detach().cpu().numpy()
                geom_case_curv.append(case_curv)
            if "torsion" in aux:
                case_tors = aux["torsion"].abs().mean(dim=1).detach().cpu().numpy()
                geom_case_tors.append(case_tors)
    risk_all = np.concatenate(risks, axis=0)
    time_all = np.concatenate(times, axis=0)
    cens_all = np.concatenate(censorships, axis=0)
    geom_stats = {
        "curvature_case_mean": float("nan"),
        "curvature_case_std": float("nan"),
        "torsion_case_mean": float("nan"),
        "torsion_case_std": float("nan"),
    }
    if geom_case_curv:
        curv_all = np.concatenate(geom_case_curv, axis=0)
        geom_stats["curvature_case_mean"] = float(np.mean(curv_all))
        geom_stats["curvature_case_std"] = float(np.std(curv_all))
    if geom_case_tors:
        tors_all = np.concatenate(geom_case_tors, axis=0)
        geom_stats["torsion_case_mean"] = float(np.mean(tors_all))
        geom_stats["torsion_case_std"] = float(np.std(tors_all))
    return concordance_index(risk_all, time_all, cens_all), float(np.mean(losses)), geom_stats


def train_once(
    *,
    groups: dict[str, list[IndexRow]],
    case_table: dict[str, dict[str, float]],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    tile_dim: int,
    max_tiles: int,
    use_multi_slide: bool,
    multi_slide_tile_budget_mode: str,
    seed: int,
    method: str,
    num_points: int,
    b_points_level: str,
    hidden_dim: int,
    point_feature_dim: int,
    fusion_mode: str,
    val_select_mode: str,
    ema_alpha: float,
    smooth_l1_weight: float,
    smooth_l2_weight: float,
    curvature_reg_weight: float,
    torsion_reg_weight: float,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
) -> dict[str, object]:
    set_seed(int(seed))

    train_ds = WSITokenCaseDataset(
        groups=groups,
        case_ids=sorted(train_ids),
        max_tiles=int(max_tiles),
        use_multi_slide=bool(use_multi_slide),
        multi_slide_tile_budget_mode=str(multi_slide_tile_budget_mode),
    )
    val_ds = WSITokenCaseDataset(
        groups=groups,
        case_ids=sorted(val_ids),
        max_tiles=int(max_tiles),
        use_multi_slide=bool(use_multi_slide),
        multi_slide_tile_budget_mode=str(multi_slide_tile_budget_mode),
    )
    test_ds = WSITokenCaseDataset(
        groups=groups,
        case_ids=sorted(test_ids),
        max_tiles=int(max_tiles),
        use_multi_slide=bool(use_multi_slide),
        multi_slide_tile_budget_mode=str(multi_slide_tile_budget_mode),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_token_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_token_samples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_token_samples,
    )

    model = TinyWSIBPointsCox(
        tile_dim=int(tile_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        method=str(method),
        num_points=int(num_points),
        b_points_level=str(b_points_level),
        point_feature_dim=int(point_feature_dim),
        fusion_mode=str(fusion_mode),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_epoch = 0
    best_val = float("-inf")
    best_val_raw = float("nan")
    best_val_ema = float("nan")
    best_test = float("nan")
    best_train = float("nan")
    best_train_geom_stats: dict[str, float] = {}
    best_val_geom_stats: dict[str, float] = {}
    best_test_geom_stats: dict[str, float] = {}
    history: list[dict[str, object]] = []
    ema_val = float("nan")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            case_ids = batch["case_id"]
            tt = batch["tile_tokens"].to(device)
            am = batch["tile_attn_mask"].to(device)
            sid = batch["slide_ids"].to(device)
            t = torch.tensor([case_table[str(c)]["event_time"] for c in case_ids], device=device, dtype=torch.float32)
            c = torch.tensor([case_table[str(c)]["censorship"] for c in case_ids], device=device, dtype=torch.float32)
            e = 1.0 - c
            risk, aux_train = model(tt, am, sid)
            loss = neg_partial_log_likelihood(risk, t, e)
            if str(method) in {"free_points", "weighted_points"}:
                loss = loss + float(smooth_l1_weight) * aux_train["smooth_l1"] + float(smooth_l2_weight) * aux_train["smooth_l2"]
                loss = loss + float(curvature_reg_weight) * aux_train["curvature_l2"] + float(torsion_reg_weight) * aux_train["torsion_l2"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        train_c, _, train_geom_stats = evaluate(model, train_loader, case_table, device)
        val_c, val_l, val_geom_stats = evaluate(model, val_loader, case_table, device)
        test_c, _, test_geom_stats = evaluate(model, test_loader, case_table, device)
        if str(val_select_mode) == "ema":
            if np.isnan(ema_val):
                ema_val = float(val_c)
            else:
                ema_val = float(ema_alpha) * float(val_c) + (1.0 - float(ema_alpha)) * float(ema_val)
            val_metric = float(ema_val)
        else:
            val_metric = float(val_c)
        history.append(
            {
                "epoch": int(epoch),
                "train_c_index": float(train_c),
                "val_c_index": float(val_c),
                "val_metric": float(val_metric),
                "val_ema": float(ema_val) if not np.isnan(ema_val) else float("nan"),
                "val_loss": float(val_l),
                "test_c_index": float(test_c),
                "train_curvature_case_mean": float(train_geom_stats.get("curvature_case_mean", float("nan"))),
                "train_curvature_case_std": float(train_geom_stats.get("curvature_case_std", float("nan"))),
                "train_torsion_case_mean": float(train_geom_stats.get("torsion_case_mean", float("nan"))),
                "train_torsion_case_std": float(train_geom_stats.get("torsion_case_std", float("nan"))),
                "val_curvature_case_mean": float(val_geom_stats.get("curvature_case_mean", float("nan"))),
                "val_curvature_case_std": float(val_geom_stats.get("curvature_case_std", float("nan"))),
                "val_torsion_case_mean": float(val_geom_stats.get("torsion_case_mean", float("nan"))),
                "val_torsion_case_std": float(val_geom_stats.get("torsion_case_std", float("nan"))),
                "test_curvature_case_mean": float(test_geom_stats.get("curvature_case_mean", float("nan"))),
                "test_curvature_case_std": float(test_geom_stats.get("curvature_case_std", float("nan"))),
                "test_torsion_case_mean": float(test_geom_stats.get("torsion_case_mean", float("nan"))),
                "test_torsion_case_std": float(test_geom_stats.get("torsion_case_std", float("nan"))),
            }
        )
        if float(val_metric) > float(best_val):
            best_val = float(val_metric)
            best_val_raw = float(val_c)
            best_val_ema = float(ema_val) if not np.isnan(ema_val) else float("nan")
            best_epoch = int(epoch)
            best_test = float(test_c)
            best_train = float(train_c)
            best_train_geom_stats = dict(train_geom_stats)
            best_val_geom_stats = dict(val_geom_stats)
            best_test_geom_stats = dict(test_geom_stats)

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_train_c_index": float(best_train),
        "best_val_c_index": float(best_val),
        "best_val_raw_c_index": float(best_val_raw),
        "best_val_ema_c_index": float(best_val_ema),
        "best_test_c_index": float(best_test),
        "best_train_curvature_case_mean": float(best_train_geom_stats.get("curvature_case_mean", float("nan"))),
        "best_train_curvature_case_std": float(best_train_geom_stats.get("curvature_case_std", float("nan"))),
        "best_train_torsion_case_mean": float(best_train_geom_stats.get("torsion_case_mean", float("nan"))),
        "best_train_torsion_case_std": float(best_train_geom_stats.get("torsion_case_std", float("nan"))),
        "best_val_curvature_case_mean": float(best_val_geom_stats.get("curvature_case_mean", float("nan"))),
        "best_val_curvature_case_std": float(best_val_geom_stats.get("curvature_case_std", float("nan"))),
        "best_val_torsion_case_mean": float(best_val_geom_stats.get("torsion_case_mean", float("nan"))),
        "best_val_torsion_case_std": float(best_val_geom_stats.get("torsion_case_std", float("nan"))),
        "best_test_curvature_case_mean": float(best_test_geom_stats.get("curvature_case_mean", float("nan"))),
        "best_test_curvature_case_std": float(best_test_geom_stats.get("curvature_case_std", float("nan"))),
        "best_test_torsion_case_mean": float(best_test_geom_stats.get("torsion_case_mean", float("nan"))),
        "best_test_torsion_case_std": float(best_test_geom_stats.get("torsion_case_std", float("nan"))),
        "train_case_count": int(len(train_ds)),
        "val_case_count": int(len(val_ds)),
        "test_case_count": int(len(test_ds)),
        "history": history,
    }


def group_rows_by_case(rows: list[IndexRow], *, source_name: str, case_table: dict[str, dict[str, float]]) -> dict[str, list[IndexRow]]:
    groups: dict[str, list[IndexRow]] = defaultdict(list)
    for r in rows:
        if r.case_id is None:
            continue
        if str(r.case_id) not in case_table:
            continue
        if str(r.wsi_feature_source) != str(source_name):
            continue
        groups[str(r.case_id)].append(r)
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description="Proxy Cox benchmark using PLIP tile tokens with optional B-points geometry branch.")
    p.add_argument("--project-root", default=str(_ROOT))
    p.add_argument("--data-root", default=None)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--target-col", default="dss_survival_days")
    p.add_argument("--wsi-feature-source", required=True)
    p.add_argument("--max-tiles", type=int, default=256)
    p.add_argument("--use-multi-slide", action="store_true")
    p.add_argument("--multi-slide-tile-budget-mode", choices=["per_slide", "case_shared"], default="per_slide")
    p.add_argument("--fixed-split", action="store_true")
    p.add_argument("--fixed-split-seed", type=int, default=0)
    p.add_argument("--fixed-train-frac", type=float, default=0.7)
    p.add_argument("--fixed-val-frac", type=float, default=0.1)
    p.add_argument("--fixed-test-frac", type=float, default=0.2)
    p.add_argument("--method", choices=["mean", "b_points", "var_interleave", "free_points", "weighted_points"], default="mean")
    p.add_argument("--b-points-level", choices=["case", "slide"], default="case")
    p.add_argument("--num-points", type=int, default=24)
    p.add_argument("--points-sweep", default=None, help="Comma separated list of num_points to sweep.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--point-feature-dim", type=int, default=24)
    p.add_argument("--point-feature-dim-sweep", default=None, help="Comma separated list of lifted per-point feature dims.")
    p.add_argument(
        "--fusion-mode",
        choices=["concat", "gated", "geo_residual", "tile_self_attn", "tile_cross_attn", "refine_cross_attn", "slide_case_attn", "guided_slide_case_attn"],
        default="concat",
    )
    p.add_argument("--fusion-mode-sweep", default=None, help="Comma separated list of geometry fusion modes.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--smooth-l1-weight", type=float, default=0.0)
    p.add_argument("--smooth-l2-weight", type=float, default=0.0)
    p.add_argument("--curvature-reg-weight", type=float, default=0.0)
    p.add_argument("--torsion-reg-weight", type=float, default=0.0)
    p.add_argument("--val-select-mode", choices=["raw", "ema"], default="raw")
    p.add_argument("--ema-alpha", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--val-time-bins", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    paths = resolve_data_paths(project_root=project_root, data_root=args.data_root)
    paths.validate()
    inventory = scan_all_assets(paths)
    rows = build_index(inventory)
    case_table, train_case_ids, test_case_ids = load_official_split(split_dir, str(args.target_col))
    groups = group_rows_by_case(rows, source_name=str(args.wsi_feature_source), case_table=case_table)
    usable_cases = sorted([c for c in groups.keys() if c in case_table])
    if not usable_cases:
        raise ValueError("no usable cases found after filtering by split and feature source")
    sample_row = sorted(groups[usable_cases[0]], key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))[0]
    sample_tokens, _, _ = load_wsi_input(
        Path(sample_row.wsi_feature_path),
        kind=str(sample_row.wsi_feature_kind),
        source_name=str(sample_row.wsi_feature_source),
        max_tiles=1,
    )
    tile_dim = int(sample_tokens.shape[1])

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if args.points_sweep is None:
        points_list = [int(args.num_points)]
    else:
        points_list = [int(x.strip()) for x in str(args.points_sweep).split(",") if x.strip()]
    if args.point_feature_dim_sweep is None:
        point_feature_dims = [int(args.point_feature_dim)]
    else:
        point_feature_dims = [int(x.strip()) for x in str(args.point_feature_dim_sweep).split(",") if x.strip()]
    if args.fusion_mode_sweep is None:
        fusion_modes = [str(args.fusion_mode)]
    else:
        fusion_modes = [str(x.strip()) for x in str(args.fusion_mode_sweep).split(",") if x.strip()]

    all_run_rows: list[dict[str, object]] = []
    meta: dict[str, object] = {
        "project_root": str(project_root),
        "split_dir": str(split_dir),
        "target_col": str(args.target_col),
        "wsi_feature_source": str(args.wsi_feature_source),
        "max_tiles": int(args.max_tiles),
        "use_multi_slide": bool(args.use_multi_slide),
        "multi_slide_tile_budget_mode": str(args.multi_slide_tile_budget_mode),
        "fixed_split": bool(args.fixed_split),
        "fixed_split_seed": int(args.fixed_split_seed),
        "fixed_train_frac": float(args.fixed_train_frac),
        "fixed_val_frac": float(args.fixed_val_frac),
        "fixed_test_frac": float(args.fixed_test_frac),
        "method": str(args.method),
        "b_points_level": str(args.b_points_level),
        "hidden_dim": int(args.hidden_dim),
        "point_feature_dims": point_feature_dims,
        "fusion_modes": fusion_modes,
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "smooth_l1_weight": float(args.smooth_l1_weight),
        "smooth_l2_weight": float(args.smooth_l2_weight),
        "curvature_reg_weight": float(args.curvature_reg_weight),
        "torsion_reg_weight": float(args.torsion_reg_weight),
        "val_select_mode": str(args.val_select_mode),
        "ema_alpha": float(args.ema_alpha),
        "val_frac": float(args.val_frac),
        "val_time_bins": int(args.val_time_bins),
        "seeds": seeds,
        "points_list": points_list,
        "device": str(device),
        "n_cases_total": int(len(usable_cases)),
    }
    save_json(out_dir / "meta.json", meta)

    fixed_split_ids: tuple[set[str], set[str], set[str]] | None = None
    if bool(args.fixed_split):
        fixed_train_ids, fixed_val_ids, fixed_test_ids = split_fixed_stratified_case_ids(
            usable_cases,
            seed=int(args.fixed_split_seed),
            train_frac=float(args.fixed_train_frac),
            val_frac=float(args.fixed_val_frac),
            test_frac=float(args.fixed_test_frac),
            case_table=case_table,
            n_time_bins=int(args.val_time_bins),
        )
        fixed_train_ids = {c for c in fixed_train_ids if c in groups}
        fixed_val_ids = {c for c in fixed_val_ids if c in groups}
        fixed_test_ids = {c for c in fixed_test_ids if c in groups}
        fixed_split_ids = (fixed_train_ids, fixed_val_ids, fixed_test_ids)
        write_csv(
            out_dir / "fixed_split_train.csv",
            [{"case_id": str(c)} for c in sorted(fixed_train_ids)],
            ["case_id"],
        )
        write_csv(
            out_dir / "fixed_split_val.csv",
            [{"case_id": str(c)} for c in sorted(fixed_val_ids)],
            ["case_id"],
        )
        write_csv(
            out_dir / "fixed_split_test.csv",
            [{"case_id": str(c)} for c in sorted(fixed_test_ids)],
            ["case_id"],
        )

    histories: dict[str, object] = {}
    for num_points in points_list:
        for point_feature_dim in point_feature_dims:
            for fusion_mode in fusion_modes:
                for seed in seeds:
                    train_ids, val_ids = split_train_val_case_ids(
                        sorted(train_case_ids),
                        int(seed),
                        float(args.val_frac),
                        case_table,
                        n_time_bins=int(args.val_time_bins),
                    )
                    if fixed_split_ids is None:
                        train_ids = {c for c in train_ids if c in groups}
                        val_ids = {c for c in val_ids if c in groups}
                        test_ids = {c for c in test_case_ids if c in groups}
                    else:
                        train_ids, val_ids, test_ids = fixed_split_ids
                    run = train_once(
                        groups=groups,
                        case_table=case_table,
                        train_ids=train_ids,
                        val_ids=val_ids,
                        test_ids=test_ids,
                        tile_dim=int(tile_dim),
                        max_tiles=int(args.max_tiles),
                        use_multi_slide=bool(args.use_multi_slide),
                        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
                        seed=int(seed),
                        method=str(args.method),
                        num_points=int(num_points),
                        b_points_level=str(args.b_points_level),
                        hidden_dim=int(args.hidden_dim),
                        point_feature_dim=int(point_feature_dim),
                        fusion_mode=str(fusion_mode),
                        val_select_mode=str(args.val_select_mode),
                        ema_alpha=float(args.ema_alpha),
                        smooth_l1_weight=float(args.smooth_l1_weight),
                        smooth_l2_weight=float(args.smooth_l2_weight),
                        curvature_reg_weight=float(args.curvature_reg_weight),
                        torsion_reg_weight=float(args.torsion_reg_weight),
                        dropout=float(args.dropout),
                        batch_size=int(args.batch_size),
                        epochs=int(args.epochs),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        device=device,
                        num_workers=int(args.num_workers),
                    )
                    key = f"method_{str(args.method)}_points_{int(num_points)}_pfdim_{int(point_feature_dim)}_fusion_{str(fusion_mode)}_seed_{int(seed)}"
                    histories[key] = run["history"]
                    all_run_rows.append(
                        {
                            "method": str(args.method),
                            "num_points": int(num_points),
                            "point_feature_dim": int(point_feature_dim),
                            "fusion_mode": str(fusion_mode),
                            "seed": int(seed),
                            "best_epoch": int(run["best_epoch"]),
                            "best_train_c_index": float(run["best_train_c_index"]),
                            "best_val_c_index": float(run["best_val_c_index"]),
                            "best_val_raw_c_index": float(run["best_val_raw_c_index"]),
                            "best_val_ema_c_index": float(run["best_val_ema_c_index"]),
                            "best_test_c_index": float(run["best_test_c_index"]),
                            "best_train_curvature_case_mean": float(run["best_train_curvature_case_mean"]),
                            "best_train_curvature_case_std": float(run["best_train_curvature_case_std"]),
                            "best_train_torsion_case_mean": float(run["best_train_torsion_case_mean"]),
                            "best_train_torsion_case_std": float(run["best_train_torsion_case_std"]),
                            "best_val_curvature_case_mean": float(run["best_val_curvature_case_mean"]),
                            "best_val_curvature_case_std": float(run["best_val_curvature_case_std"]),
                            "best_val_torsion_case_mean": float(run["best_val_torsion_case_mean"]),
                            "best_val_torsion_case_std": float(run["best_val_torsion_case_std"]),
                            "best_test_curvature_case_mean": float(run["best_test_curvature_case_mean"]),
                            "best_test_curvature_case_std": float(run["best_test_curvature_case_std"]),
                            "best_test_torsion_case_mean": float(run["best_test_torsion_case_mean"]),
                            "best_test_torsion_case_std": float(run["best_test_torsion_case_std"]),
                            "train_case_count": int(run["train_case_count"]),
                            "val_case_count": int(run["val_case_count"]),
                            "test_case_count": int(run["test_case_count"]),
                        }
                    )

    write_csv(
        out_dir / "proxy_run_results.csv",
        all_run_rows,
        list(all_run_rows[0].keys()) if all_run_rows else ["num_points"],
    )
    save_json(out_dir / "histories.json", histories)
    print(f"proxy_run_results_csv={out_dir / 'proxy_run_results.csv'}", flush=True)
    print(f"histories_json={out_dir / 'histories.json'}", flush=True)
    print(f"meta_json={out_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
