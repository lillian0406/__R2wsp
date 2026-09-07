from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Batch:
    case_id: list[str]
    cohort: list[str]
    tile_tokens: torch.Tensor
    tile_xy: torch.Tensor
    tile_attn_mask: torch.Tensor
    slide_ids: torch.Tensor
    rna_vec: torch.Tensor | None
    rna_omics: list[torch.Tensor] | None
    wsi_anti_vec: torch.Tensor | None = None
    rna_anti_vec: torch.Tensor | None = None


def collate_tiles_and_rna(samples: list[dict[str, object]]) -> Batch:
    if not samples:
        raise ValueError("empty batch")

    case_ids = [str(s["case_id"]) for s in samples]
    cohorts = [str(s["cohort"]) for s in samples]

    if "rna_omics" in samples[0]:
        omics0 = samples[0]["rna_omics"]
        if not isinstance(omics0, list):
            raise ValueError("rna_omics must be a list")
        p = len(omics0)
        rna_omics: list[torch.Tensor] = []
        for i in range(p):
            rna_omics.append(torch.stack([s["rna_omics"][i] for s in samples], dim=0))
        rna_vec = None
    else:
        rna_vec = torch.stack([s["rna_vec"] for s in samples], dim=0)
        rna_omics = None

    max_n = max(int(s["tile_tokens"].shape[0]) for s in samples)
    d = int(samples[0]["tile_tokens"].shape[1])
    tile_tokens = torch.zeros((len(samples), max_n, d), dtype=torch.float32)
    tile_xy = torch.zeros((len(samples), max_n, 2), dtype=torch.float32)
    tile_attn_mask = torch.zeros((len(samples), max_n), dtype=torch.long)
    slide_ids = torch.full((len(samples), max_n), fill_value=-1, dtype=torch.long)

    for i, s in enumerate(samples):
        tt = s["tile_tokens"]
        xy = s["tile_xy"]
        am = s["tile_attn_mask"]
        sid = s.get("slide_ids")
        n = int(tt.shape[0])
        if int(tt.shape[1]) != d:
            raise ValueError("inconsistent tile embedding dim")
        tile_tokens[i, :n] = tt
        tile_xy[i, :n] = xy
        tile_attn_mask[i, :n] = am
        if sid is None:
            slide_ids[i, :n] = 0
        else:
            slide_ids[i, :n] = sid

    if "wsi_anti_vec" in samples[0] and "rna_anti_vec" in samples[0]:
        wsi_anti = torch.stack([torch.as_tensor(s["wsi_anti_vec"], dtype=torch.float32) for s in samples], dim=0)
        rna_anti = torch.stack([torch.as_tensor(s["rna_anti_vec"], dtype=torch.float32) for s in samples], dim=0)
    else:
        wsi_anti = None
        rna_anti = None

    return Batch(
        case_id=case_ids,
        cohort=cohorts,
        tile_tokens=tile_tokens,
        tile_xy=tile_xy,
        tile_attn_mask=tile_attn_mask,
        slide_ids=slide_ids,
        rna_vec=rna_vec,
        rna_omics=rna_omics,
        wsi_anti_vec=wsi_anti,
        rna_anti_vec=rna_anti,
    )
