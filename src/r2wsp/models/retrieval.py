from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(dtype=torch.float32).unsqueeze(-1)
    s = (x * w).sum(dim=1)
    d = w.sum(dim=1).clamp_min(1.0)
    return s / d


def _infonce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    logits = (a @ b.t()) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = nn.functional.cross_entropy(logits, labels)
    loss_b = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


@dataclass(frozen=True)
class RetrievalOutputs:
    wsi_case: torch.Tensor
    rna_case: torch.Tensor
    r2w_case: torch.Tensor
    w2r_case: torch.Tensor


class R2wspRetrieval(nn.Module):
    def __init__(
        self,
        *,
        tile_dim: int = 512,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_rna_tokens: int = 8,
        rna_encoder: str = "mean",
        omic_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_rna_tokens = int(num_rna_tokens)
        self.rna_encoder = str(rna_encoder)
        if self.rna_encoder not in {"mean", "omics_mlp"}:
            raise ValueError("rna_encoder must be mean|omics_mlp")

        self.wsi_proj = nn.Linear(int(tile_dim), int(dim))

        if self.rna_encoder == "mean":
            self.rna_to_tokens = nn.Sequential(
                nn.Linear(1, int(dim)),
                nn.GELU(),
                nn.Linear(int(dim), int(dim) * int(num_rna_tokens)),
            )
            self.rna_sig_networks: nn.ModuleList | None = None
        else:
            if not omic_sizes:
                raise ValueError("omic_sizes is required when rna_encoder=omics_mlp")
            self.rna_to_tokens = None
            self.rna_sig_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(s), int(dim)),
                        nn.ELU(),
                        nn.AlphaDropout(float(dropout)),
                        nn.Linear(int(dim), int(dim)),
                        nn.ELU(),
                        nn.AlphaDropout(float(dropout)),
                    )
                    for s in omic_sizes
                ]
            )

        self.r2w_attn = nn.MultiheadAttention(int(dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.w2r_attn = nn.MultiheadAttention(int(dim), int(num_heads), dropout=float(dropout), batch_first=True)

        self.dropout = nn.Dropout(float(dropout))

    def _rna_tokens_from_vec(self, rna_vec: torch.Tensor) -> torch.Tensor:
        rna_mean = rna_vec.mean(dim=1, keepdim=True)
        t = self.rna_to_tokens(rna_mean)
        return t.view(rna_vec.shape[0], int(self.num_rna_tokens), int(self.dim))

    def forward(
        self,
        *,
        tile_tokens: torch.Tensor,
        tile_attn_mask: torch.Tensor,
        rna_vec: torch.Tensor | None = None,
        rna_omics: list[torch.Tensor] | None = None,
    ) -> RetrievalOutputs:
        wsi_tokens = self.dropout(self.wsi_proj(tile_tokens))
        wsi_case = _masked_mean(wsi_tokens, tile_attn_mask)

        if self.rna_encoder == "mean":
            if rna_vec is None:
                raise ValueError("rna_vec is required when rna_encoder=mean")
            rna_tokens = self._rna_tokens_from_vec(rna_vec)
        else:
            if rna_omics is None:
                raise ValueError("rna_omics is required when rna_encoder=omics_mlp")
            if self.rna_sig_networks is None:
                raise RuntimeError("rna_sig_networks not initialized")
            if len(rna_omics) != len(self.rna_sig_networks):
                raise ValueError("rna_omics length mismatch")
            feats = [net(x.float()) for net, x in zip(self.rna_sig_networks, rna_omics, strict=True)]
            rna_tokens = torch.stack(feats, dim=1)
        rna_case = rna_tokens.mean(dim=1)

        key_padding_mask = tile_attn_mask <= 0

        r2w_tokens, _ = self.r2w_attn(
            query=rna_tokens,
            key=wsi_tokens,
            value=wsi_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        r2w_case = r2w_tokens.mean(dim=1)

        w2r_tokens, _ = self.w2r_attn(
            query=wsi_tokens,
            key=rna_tokens,
            value=rna_tokens,
            need_weights=False,
        )
        w2r_case = _masked_mean(w2r_tokens, tile_attn_mask)

        return RetrievalOutputs(wsi_case=wsi_case, rna_case=rna_case, r2w_case=r2w_case, w2r_case=w2r_case)

    def compute_loss(self, out: RetrievalOutputs, *, temperature: float = 0.07) -> dict[str, torch.Tensor]:
        loss_align = _infonce(out.wsi_case, out.rna_case, temperature=float(temperature))
        loss_r2w = _infonce(out.r2w_case, out.rna_case, temperature=float(temperature))
        loss_w2r = _infonce(out.wsi_case, out.w2r_case, temperature=float(temperature))
        loss = loss_align + 0.5 * (loss_r2w + loss_w2r)
        return {
            "loss": loss,
            "loss_align": loss_align.detach(),
            "loss_r2w": loss_r2w.detach(),
            "loss_w2r": loss_w2r.detach(),
        }
