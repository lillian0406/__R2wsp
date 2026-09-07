from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import openslide
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


@dataclass(frozen=True)
class TilePlan:
    level: int
    read_size_level: int
    span_level0: int
    stride_level0: int
    slide_mpp: float
    target_mpp: float


def sanitize_feature_name(text: str) -> str:
    out = str(text).strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    out = out.replace(" ", "_")
    return out or "features"


def infer_slide_mpp(slide: openslide.OpenSlide, *, fallback_mpp: float = 0.5) -> float:
    props = slide.properties
    for key in (
        openslide.PROPERTY_NAME_MPP_X,
        "aperio.MPP",
        "openslide.mpp-x",
        "mpp_x",
    ):
        value = props.get(key)
        if value is None:
            continue
        try:
            mpp = float(value)
        except (TypeError, ValueError):
            continue
        if mpp > 0:
            return mpp

    objective_power = None
    for key in (
        openslide.PROPERTY_NAME_OBJECTIVE_POWER,
        "aperio.AppMag",
        "objective_power",
    ):
        value = props.get(key)
        if value is None:
            continue
        try:
            objective_power = float(value)
        except (TypeError, ValueError):
            objective_power = None
        if objective_power is not None:
            break

    if objective_power is not None:
        if objective_power >= 39:
            return 0.25
        if objective_power >= 19:
            return 0.5

    return float(fallback_mpp)


def build_tile_plan(
    slide: openslide.OpenSlide,
    *,
    target_mpp: float,
    tile_px: int,
    stride_px: int | None = None,
    fallback_slide_mpp: float = 0.5,
) -> TilePlan:
    slide_mpp = infer_slide_mpp(slide, fallback_mpp=fallback_slide_mpp)
    target_downsample = max(float(target_mpp) / float(slide_mpp), 1.0)
    level_downsamples = [float(x) for x in slide.level_downsamples]
    level = min(
        range(len(level_downsamples)),
        key=lambda idx: abs(math.log(level_downsamples[idx]) - math.log(target_downsample)),
    )
    chosen_downsample = level_downsamples[level]
    read_size_level = max(1, int(round(float(tile_px) * target_downsample / chosen_downsample)))
    span_level0 = max(1, int(round(float(tile_px) * target_downsample)))
    stride_level0 = max(1, int(round(float(stride_px or tile_px) * target_downsample)))
    return TilePlan(
        level=level,
        read_size_level=read_size_level,
        span_level0=span_level0,
        stride_level0=stride_level0,
        slide_mpp=float(slide_mpp),
        target_mpp=float(target_mpp),
    )


def build_tissue_mask(
    slide: openslide.OpenSlide,
    *,
    max_thumbnail_size: int = 2048,
    white_threshold: float = 0.82,
    saturation_threshold: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    width, height = slide.dimensions
    scale = min(1.0, float(max_thumbnail_size) / float(max(width, height)))
    thumb_w = max(1, int(round(width * scale)))
    thumb_h = max(1, int(round(height * scale)))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = maxc - minc
    val = maxc
    tissue = (val < float(white_threshold)) & (sat > float(saturation_threshold))
    return tissue.astype(np.uint8), float(width) / float(thumb_w), float(height) / float(thumb_h)


def generate_tile_coords(
    slide: openslide.OpenSlide,
    plan: TilePlan,
    *,
    tissue_mask: np.ndarray,
    thumb_scale_x: float,
    thumb_scale_y: float,
    min_tissue_fraction: float = 0.2,
    max_tiles: int | None = None,
    edge: bool = False,
) -> np.ndarray:
    width, height = slide.dimensions
    span = int(plan.span_level0)
    stride = int(plan.stride_level0)
    if edge:
        xs = list(range(0, max(1, width - span + 1), stride))
        ys = list(range(0, max(1, height - span + 1), stride))
        if xs[-1] != max(0, width - span):
            xs.append(max(0, width - span))
        if ys[-1] != max(0, height - span):
            ys.append(max(0, height - span))
    else:
        xs = list(range(0, max(0, width - span + 1), stride))
        ys = list(range(0, max(0, height - span + 1), stride))

    candidates: list[tuple[float, int, int]] = []
    mask_h, mask_w = tissue_mask.shape
    for y in ys:
        y0 = max(0, min(mask_h - 1, int(y / thumb_scale_y)))
        y1 = max(y0 + 1, min(mask_h, int(math.ceil((y + span) / thumb_scale_y))))
        for x in xs:
            x0 = max(0, min(mask_w - 1, int(x / thumb_scale_x)))
            x1 = max(x0 + 1, min(mask_w, int(math.ceil((x + span) / thumb_scale_x))))
            frac = float(tissue_mask[y0:y1, x0:x1].mean())
            if frac >= float(min_tissue_fraction):
                candidates.append((frac, int(x), int(y)))

    if not candidates:
        return np.zeros((0, 2), dtype=np.int64)

    if max_tiles is not None and int(max_tiles) > 0 and len(candidates) > int(max_tiles):
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: int(max_tiles)]

    coords = np.asarray([(x, y) for _, x, y in candidates], dtype=np.int64)
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    return coords[order]


class TimmFeatureExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        batch_size: int = 64,
        amp: bool = True,
    ) -> None:
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.amp = bool(amp and self.device.type == "cuda")
        checkpoint_arg = str(Path(checkpoint_path).resolve()) if checkpoint_path else ""
        self.model = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
            checkpoint_path=checkpoint_arg,
        )
        self.model.eval().to(self.device)
        self.data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**self.data_config, is_training=False)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    output = self.model(batch)
            else:
                output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                output = output["x_norm_clstoken"]
            elif "pooler_output" in output:
                output = output["pooler_output"]
            else:
                output = next(iter(output.values()))
        if output.ndim == 3:
            output = output[:, 0, :]
        if output.ndim != 2:
            raise ValueError(f"expected 2D features, got shape={tuple(output.shape)}")
        return output.to(dtype=torch.float32)

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        feats: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            batch = torch.stack([self.transform(img) for img in batch_images], dim=0).to(self.device, non_blocking=True)
            output = self._forward(batch).detach().cpu()
            feats.append(output)
        return torch.cat(feats, dim=0).numpy().astype(np.float32, copy=False)


def extract_slide_features_to_h5(
    *,
    wsi_path: str | Path,
    output_h5: str | Path,
    extractor: TimmFeatureExtractor,
    tile_px: int = 256,
    target_mpp: float = 0.5,
    stride_px: int | None = None,
    max_thumbnail_size: int = 2048,
    min_tissue_fraction: float = 0.2,
    max_tiles: int | None = None,
    edge: bool = False,
    fallback_slide_mpp: float = 0.5,
    overwrite: bool = False,
    progress_every_batches: int = 0,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    wsi_path = Path(wsi_path).resolve()
    output_h5 = Path(output_h5).resolve()
    partial_h5 = output_h5.with_suffix(output_h5.suffix + ".partial")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists() and not overwrite:
        return {"status": "skipped", "wsi_path": str(wsi_path), "output_h5": str(output_h5)}
    if partial_h5.exists():
        partial_h5.unlink()

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        plan = build_tile_plan(
            slide,
            target_mpp=float(target_mpp),
            tile_px=int(tile_px),
            stride_px=stride_px,
            fallback_slide_mpp=float(fallback_slide_mpp),
        )
        tissue_mask, thumb_scale_x, thumb_scale_y = build_tissue_mask(slide, max_thumbnail_size=int(max_thumbnail_size))
        coords = generate_tile_coords(
            slide,
            plan,
            tissue_mask=tissue_mask,
            thumb_scale_x=thumb_scale_x,
            thumb_scale_y=thumb_scale_y,
            min_tissue_fraction=float(min_tissue_fraction),
            max_tiles=max_tiles,
            edge=bool(edge),
        )
        if coords.shape[0] == 0:
            raise RuntimeError(f"no tissue tiles selected for {wsi_path}")

        batch_size = max(1, int(extractor.batch_size))
        total_batches = (coords.shape[0] + batch_size - 1) // batch_size
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "slide_start",
                    "slide_id": wsi_path.stem,
                    "wsi_path": str(wsi_path),
                    "output_h5": str(output_h5),
                    "partial_h5": str(partial_h5),
                    "n_tiles": int(coords.shape[0]),
                    "total_batches": int(total_batches),
                    "feat_dim": None,
                }
            )

        n_written = 0
        feat_dim: int | None = None
        with h5py.File(partial_h5, "w") as handle:
            features_ds = None
            coords_ds = None
            for batch_idx, start in enumerate(range(0, coords.shape[0], batch_size), start=1):
                images: list[Image.Image] = []
                batch_coords = coords[start : start + batch_size]
                for x, y in batch_coords.tolist():
                    region = slide.read_region((int(x), int(y)), plan.level, (plan.read_size_level, plan.read_size_level)).convert("RGB")
                    if region.size != (int(tile_px), int(tile_px)):
                        region = region.resize((int(tile_px), int(tile_px)), resample=Image.BILINEAR)
                    images.append(region)
                batch_features = extractor.encode_pil_images(images)
                if feat_dim is None:
                    feat_dim = int(batch_features.shape[1])
                    features_ds = handle.create_dataset(
                        "features",
                        shape=(0, feat_dim),
                        maxshape=(None, feat_dim),
                        dtype=np.float32,
                        compression="gzip",
                    )
                    coords_ds = handle.create_dataset(
                        "coords",
                        shape=(0, 2),
                        maxshape=(None, 2),
                        dtype=np.int64,
                        compression="gzip",
                    )
                assert features_ds is not None
                assert coords_ds is not None
                next_n = n_written + batch_features.shape[0]
                features_ds.resize((next_n, feat_dim))
                coords_ds.resize((next_n, 2))
                features_ds[n_written:next_n] = batch_features
                coords_ds[n_written:next_n] = batch_coords.astype(np.int64, copy=False)
                n_written = next_n
                if progress_callback is not None and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or (int(progress_every_batches) > 0 and batch_idx % int(progress_every_batches) == 0)
                ):
                    progress_callback(
                        {
                            "event": "slide_progress",
                            "slide_id": wsi_path.stem,
                            "wsi_path": str(wsi_path),
                            "output_h5": str(output_h5),
                            "partial_h5": str(partial_h5),
                            "batch_idx": int(batch_idx),
                            "total_batches": int(total_batches),
                            "tiles_written": int(n_written),
                            "n_tiles": int(coords.shape[0]),
                            "feat_dim": int(feat_dim),
                        }
                    )

            if n_written != coords.shape[0]:
                raise RuntimeError(f"feature/coord mismatch: {n_written} vs {coords.shape[0]}")

            handle.attrs["slide_id"] = wsi_path.stem
            handle.attrs["model_name"] = extractor.model_name
            handle.attrs["target_mpp"] = float(plan.target_mpp)
            handle.attrs["slide_mpp"] = float(plan.slide_mpp)
            handle.attrs["tile_px"] = int(tile_px)
            handle.attrs["stride_px"] = int(stride_px or tile_px)
            handle.attrs["read_level"] = int(plan.level)
            handle.attrs["read_size_level"] = int(plan.read_size_level)
            handle.attrs["span_level0"] = int(plan.span_level0)
        partial_h5.replace(output_h5)

        return {
            "status": "written",
            "wsi_path": str(wsi_path),
            "output_h5": str(output_h5),
            "n_tiles": int(coords.shape[0]),
            "feat_dim": int(feat_dim or 0),
            "slide_mpp": float(plan.slide_mpp),
            "target_mpp": float(plan.target_mpp),
        }
    finally:
        slide.close()
