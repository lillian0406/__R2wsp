import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.models.direct_survival import DirectWSIRNASurvival


def main() -> None:
    batch_size = 2
    num_tiles = 17
    tile_dim = 512
    rna_dim = 19962

    model = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=128,
        dropout=0.0,
        model_variant="custom",
        wsi_geo_type="b_points",
        wsi_geo_num_points=24,
        wsi_geo_output="points",
        wsi_geo_position="after_mean",
        wsi_geo_fusion="concat",
        rna_geo_type="none",
    ).eval()

    tile_tokens = torch.randn(batch_size, num_tiles, tile_dim)
    tile_xy = torch.zeros(batch_size, num_tiles, 2)
    tile_attn_mask = torch.ones(batch_size, num_tiles, dtype=torch.long)
    rna_vec = torch.randn(batch_size, rna_dim)

    with torch.no_grad():
        risk, aux = model(
            tile_tokens=tile_tokens,
            tile_xy=tile_xy,
            tile_attn_mask=tile_attn_mask,
            slide_ids=None,
            rna_vec=rna_vec,
        )

    points = aux["wsi_b_points_points"]
    token_attn = aux["wsi_b_points_token_attn"]

    if risk.shape != (batch_size,):
        raise RuntimeError(f"unexpected risk shape: {risk.shape}")
    if points.shape != (batch_size, 24, 3):
        raise RuntimeError(f"unexpected points shape: {points.shape}")
    if token_attn.shape != (batch_size, 24, num_tiles):
        raise RuntimeError(f"unexpected token_attn shape: {token_attn.shape}")

    model2 = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=128,
        dropout=0.0,
        model_variant="baseline",
        wsi_geo_type="b_points",
        wsi_geo_num_points=24,
        wsi_geo_output="points",
        wsi_geo_position="after_mean",
        wsi_geo_fusion="concat",
        wsi_b_points_level="case",
        rna_geo_type="none",
    ).eval()

    with torch.no_grad():
        risk2, aux2 = model2(
            tile_tokens=tile_tokens,
            tile_xy=tile_xy,
            tile_attn_mask=tile_attn_mask,
            slide_ids=torch.zeros(batch_size, num_tiles, dtype=torch.long),
            rna_vec=rna_vec,
        )

    if risk2.shape != (batch_size,):
        raise RuntimeError(f"unexpected risk2 shape: {risk2.shape}")
    if aux2["wsi_b_points_points"].shape != (batch_size, 24, 3):
        raise RuntimeError(f"unexpected baseline points shape: {aux2['wsi_b_points_points'].shape}")
    if aux2["wsi_b_points_token_attn"].shape != (batch_size, 24, num_tiles):
        raise RuntimeError(f"unexpected baseline token_attn shape: {aux2['wsi_b_points_token_attn'].shape}")

    num_tiles2 = 24
    tile_tokens2 = torch.randn(batch_size, num_tiles2, tile_dim)
    tile_xy2 = torch.zeros(batch_size, num_tiles2, 2)
    tile_attn_mask2 = torch.ones(batch_size, num_tiles2, dtype=torch.long)
    slide_ids2 = torch.cat(
        [
            torch.zeros(batch_size, num_tiles2 // 2, dtype=torch.long),
            torch.ones(batch_size, num_tiles2 - num_tiles2 // 2, dtype=torch.long),
        ],
        dim=1,
    )
    model3 = DirectWSIRNASurvival(
        tile_dim=tile_dim,
        rna_dim=rna_dim,
        hidden_dim=128,
        dropout=0.0,
        model_variant="baseline",
        use_multi_slide=True,
        multi_slide_mode="slide_attn_case_attn",
        wsi_geo_type="b_points",
        wsi_geo_num_points=24,
        wsi_geo_output="all",
        wsi_geo_position="after_mean",
        wsi_geo_fusion="concat",
        wsi_b_points_level="slide",
        rna_geo_type="none",
    ).eval()

    with torch.no_grad():
        risk3, aux3 = model3(
            tile_tokens=tile_tokens2,
            tile_xy=tile_xy2,
            tile_attn_mask=tile_attn_mask2,
            slide_ids=slide_ids2,
            rna_vec=rna_vec,
        )

    if risk3.shape != (batch_size,):
        raise RuntimeError(f"unexpected risk3 shape: {risk3.shape}")
    if aux3["wsi_b_points_points_by_slide"].shape != (batch_size, 2, 24, 3):
        raise RuntimeError(f"unexpected points_by_slide shape: {aux3['wsi_b_points_points_by_slide'].shape}")
    if aux3["wsi_b_points_points"].shape != (batch_size, 24, 3):
        raise RuntimeError(f"unexpected aggregated points shape: {aux3['wsi_b_points_points'].shape}")


if __name__ == "__main__":
    main()
