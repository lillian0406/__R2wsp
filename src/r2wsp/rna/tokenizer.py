from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .gene_sets import GeneSetCollection


@dataclass(frozen=True)
class OmicsSpec:
    pathway_names: list[str]
    gene_indices: list[torch.Tensor]

    @property
    def pathway_count(self) -> int:
        return len(self.pathway_names)

    @property
    def omic_sizes(self) -> list[int]:
        return [int(x.numel()) for x in self.gene_indices]


def build_omics_spec(genes: Iterable[str], gene_sets: GeneSetCollection) -> OmicsSpec:
    gene_to_idx = {str(g): i for i, g in enumerate(genes)}
    pathway_names: list[str] = []
    indices: list[torch.Tensor] = []
    for name, gs in zip(gene_sets.names, gene_sets.gene_sets, strict=True):
        idxs = [gene_to_idx[g] for g in gs if g in gene_to_idx]
        if not idxs:
            continue
        pathway_names.append(str(name))
        indices.append(torch.tensor(idxs, dtype=torch.long))
    if not pathway_names:
        raise ValueError("no pathway genes intersect with RNA gene list")
    return OmicsSpec(pathway_names=pathway_names, gene_indices=indices)


def tokenize_omics(rna_vec: torch.Tensor, spec: OmicsSpec) -> list[torch.Tensor]:
    if rna_vec.ndim != 1:
        raise ValueError("rna_vec must be 1D")
    out: list[torch.Tensor] = []
    for idx in spec.gene_indices:
        out.append(rna_vec.index_select(0, idx.to(device=rna_vec.device)))
    return out

