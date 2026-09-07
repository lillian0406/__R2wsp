from __future__ import annotations

from pathlib import Path

import hashlib
import numpy as np
import torch


def _stable_int_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _resolve_rng(path: Path, tile_sampling_seed: int | None) -> np.random.Generator:
    base = int(tile_sampling_seed or 0)
    mixed = (base + _stable_int_from_text(str(path))) % (2**32)
    return np.random.default_rng(mixed)


def _select_tile_indices(n: int, max_tiles: int, *, sampling: str, rng: np.random.Generator) -> np.ndarray:
    if sampling == "prefix":
        return np.arange(int(max_tiles), dtype=np.int64)
    if sampling == "random":
        idx = rng.choice(int(n), size=int(max_tiles), replace=False)
        return np.sort(np.asarray(idx, dtype=np.int64))
    raise ValueError(f"unsupported tile sampling: {sampling}")


def _finalize_arrays(feat: np.ndarray, xy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if feat.ndim != 2:
        raise ValueError("WSI features must be [N, D]")
    if xy.shape != (feat.shape[0], 2):
        raise ValueError("WSI coordinates must be [N, 2] aligned to features")
    attn_mask = np.ones((feat.shape[0],), dtype=np.int64)
    return (
        torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(dtype=torch.float32),
        torch.from_numpy(np.asarray(xy, dtype=np.float32)).to(dtype=torch.float32),
        torch.from_numpy(attn_mask).to(dtype=torch.long),
    )


def _load_npz_tokens(
    path: Path,
    max_tiles: int | None,
    *,
    tile_sampling: str,
    tile_sampling_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(path, allow_pickle=False)
    if "feat" not in data or "xy" not in data:
        raise ValueError("tile token npz must contain keys: feat, xy")
    feat = np.asarray(data["feat"], dtype=np.float32)
    xy = np.asarray(data["xy"], dtype=np.float32)
    if max_tiles is not None and feat.shape[0] > int(max_tiles):
        rng = _resolve_rng(path, tile_sampling_seed)
        idx = _select_tile_indices(int(feat.shape[0]), int(max_tiles), sampling=str(tile_sampling), rng=rng)
        feat = feat[idx]
        xy = xy[idx]
    return _finalize_arrays(feat, xy)


def _load_pt_features(
    path: Path,
    max_tiles: int | None,
    *,
    tile_sampling: str,
    tile_sampling_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        feat = obj.detach().cpu().numpy()
        xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
    elif isinstance(obj, dict):
        feat_value = obj.get("feat")
        if feat_value is None:
            feat_value = obj.get("features")
        if feat_value is None:
            raise ValueError("pt feature dict must contain feat or features")
        feat = feat_value.detach().cpu().numpy() if torch.is_tensor(feat_value) else np.asarray(feat_value, dtype=np.float32)
        xy_value = obj.get("xy")
        if xy_value is None:
            xy_value = obj.get("coords")
        if xy_value is None:
            xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
        else:
            xy = xy_value.detach().cpu().numpy() if torch.is_tensor(xy_value) else np.asarray(xy_value, dtype=np.float32)
    else:
        raise ValueError(f"unsupported pt feature payload: {type(obj).__name__}")
    feat = np.asarray(feat, dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32)
    if max_tiles is not None and feat.shape[0] > int(max_tiles):
        rng = _resolve_rng(path, tile_sampling_seed)
        idx = _select_tile_indices(int(feat.shape[0]), int(max_tiles), sampling=str(tile_sampling), rng=rng)
        feat = feat[idx]
        xy = xy[idx]
    return _finalize_arrays(feat, xy)


def _load_h5_features(
    path: Path,
    max_tiles: int | None,
    *,
    tile_sampling: str,
    tile_sampling_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import h5py

    with h5py.File(path, "r") as f:
        feat_ds = None
        for key in ("feat", "features"):
            if key in f:
                feat_ds = f[key]
                break
        if feat_ds is None:
            raise ValueError("h5 feature file must contain feat or features")
        xy_ds = None
        for key in ("xy", "coords"):
            if key in f:
                xy_ds = f[key]
                break
        total_n = int(feat_ds.shape[0])
        if max_tiles is None or total_n <= int(max_tiles):
            feat = np.asarray(feat_ds[:], dtype=np.float32)
            if xy_ds is None:
                xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
            else:
                xy = np.asarray(xy_ds[:], dtype=np.float32)
        else:
            if str(tile_sampling) == "prefix":
                n = min(total_n, int(max_tiles))
                feat = np.asarray(feat_ds[:n], dtype=np.float32)
                if xy_ds is None:
                    xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
                else:
                    xy = np.asarray(xy_ds[:n], dtype=np.float32)
            elif str(tile_sampling) == "random":
                rng = _resolve_rng(path, tile_sampling_seed)
                idx = _select_tile_indices(total_n, int(max_tiles), sampling=str(tile_sampling), rng=rng)
                feat = np.asarray(feat_ds[idx], dtype=np.float32)
                if xy_ds is None:
                    xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
                else:
                    xy = np.asarray(xy_ds[idx], dtype=np.float32)
            else:
                raise ValueError(f"unsupported tile sampling: {tile_sampling}")
    return _finalize_arrays(feat, xy)


def infer_wsi_feature_kind(path: Path, source_name: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz_tokens"
    if suffix in {".pt", ".pth"}:
        return "pt_features"
    if suffix in {".h5", ".hdf5"}:
        return "h5_features"
    if source_name is not None and "plip" in str(source_name).lower():
        return "npz_tokens"
    raise ValueError(f"unable to infer WSI feature kind from path: {path}")


def load_wsi_input(
    path: str | Path,
    *,
    kind: str | None = None,
    source_name: str | None = None,
    max_tiles: int | None = None,
    tile_sampling: str = "prefix",
    tile_sampling_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(str(feature_path))
    resolved_kind = kind or infer_wsi_feature_kind(feature_path, source_name)
    if resolved_kind == "npz_tokens":
        return _load_npz_tokens(feature_path, max_tiles, tile_sampling=tile_sampling, tile_sampling_seed=tile_sampling_seed)
    if resolved_kind == "pt_features":
        return _load_pt_features(feature_path, max_tiles, tile_sampling=tile_sampling, tile_sampling_seed=tile_sampling_seed)
    if resolved_kind == "h5_features":
        return _load_h5_features(feature_path, max_tiles, tile_sampling=tile_sampling, tile_sampling_seed=tile_sampling_seed)
    raise ValueError(f"unsupported WSI feature kind: {resolved_kind}")


def peek_wsi_num_tiles(path: str | Path, *, kind: str | None = None, source_name: str | None = None) -> int:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(str(feature_path))
    resolved_kind = kind or infer_wsi_feature_kind(feature_path, source_name)
    if resolved_kind == "npz_tokens":
        data = np.load(feature_path, allow_pickle=False)
        if "feat" not in data:
            raise ValueError("tile token npz must contain key: feat")
        return int(data["feat"].shape[0])
    if resolved_kind == "pt_features":
        obj = torch.load(feature_path, map_location="cpu", weights_only=False)
        if torch.is_tensor(obj):
            return int(obj.shape[0])
        if isinstance(obj, dict):
            feat_value = obj.get("feat")
            if feat_value is None:
                feat_value = obj.get("features")
            if feat_value is None:
                raise ValueError("pt feature dict must contain feat or features")
            return int(feat_value.shape[0]) if torch.is_tensor(feat_value) else int(np.asarray(feat_value).shape[0])
        raise ValueError(f"unsupported pt feature payload: {type(obj).__name__}")
    if resolved_kind == "h5_features":
        import h5py

        with h5py.File(feature_path, "r") as f:
            for key in ("feat", "features"):
                if key in f:
                    return int(f[key].shape[0])
        raise ValueError("h5 feature file must contain feat or features")
    raise ValueError(f"unsupported WSI feature kind: {resolved_kind}")
