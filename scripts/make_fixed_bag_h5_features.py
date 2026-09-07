from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _default_out_root(source_root: Path, target_bag_size: int) -> Path:
    if source_root.name == "feats_h5":
        tag = source_root.parent.name
        return source_root.parent.parent / f"{tag}_fixedbag{int(target_bag_size)}" / "feats_h5"
    return source_root.parent / f"{source_root.name}_fixedbag{int(target_bag_size)}"


def _seed_for_slide(slide_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{slide_id}:{int(base_seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _select_indices(n_items: int, target_bag_size: int, *, slide_id: str, base_seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(_seed_for_slide(slide_id, base_seed))
    if n_items == target_bag_size:
        return np.arange(n_items, dtype=np.int64), "keep"
    if n_items > target_bag_size:
        ids = np.sort(rng.choice(n_items, size=target_bag_size, replace=False).astype(np.int64))
        return ids, "downsample"
    reps = rng.choice(n_items, size=target_bag_size - n_items, replace=True).astype(np.int64)
    ids = np.concatenate([np.arange(n_items, dtype=np.int64), reps], axis=0)
    return ids, "upsample"


def main() -> None:
    p = argparse.ArgumentParser(description="Convert variable-length MMP-style feats_h5 into deterministic fixed-bag feats_h5.")
    p.add_argument("--source-root", required=True, help="Source feats_h5 directory.")
    p.add_argument("--out-root", default=None, help="Output feats_h5 directory.")
    p.add_argument("--target-bag-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    source_root = Path(args.source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    out_root = Path(args.out_root).resolve() if args.out_root else _default_out_root(source_root, int(args.target_bag_size)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(source_root.glob("*.h5"))
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"no .h5 files under {source_root}")

    summary: dict[str, object] = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "target_bag_size": int(args.target_bag_size),
        "seed": int(args.seed),
        "written": 0,
        "skipped": 0,
        "modes": {"keep": 0, "downsample": 0, "upsample": 0},
        "files": [],
    }

    for src_path in files:
        dst_path = out_root / src_path.name
        if dst_path.exists() and not args.overwrite:
            summary["skipped"] += 1
            continue

        with h5py.File(src_path, "r") as src:
            features = src["features"][:]
            coords = src["coords"][:]
            attrs = {k: src.attrs[k] for k in src.attrs.keys()}

        n_items = int(features.shape[0])
        feat_dim = int(features.shape[1])
        slide_id = str(attrs.get("slide_id", src_path.stem))
        ids, mode = _select_indices(n_items, int(args.target_bag_size), slide_id=slide_id, base_seed=int(args.seed))
        selected_features = features[ids].astype(np.float32, copy=False)
        selected_coords = coords[ids].astype(np.int64, copy=False)

        with h5py.File(dst_path, "w") as dst:
            dst.create_dataset("features", data=selected_features, dtype=np.float32, compression="gzip")
            dst.create_dataset("coords", data=selected_coords, dtype=np.int64, compression="gzip")
            for key, value in attrs.items():
                dst.attrs[key] = value
            dst.attrs["original_n_tiles"] = int(n_items)
            dst.attrs["fixed_bag_size"] = int(args.target_bag_size)
            dst.attrs["fixed_bag_mode"] = mode
            dst.attrs["fixed_bag_seed"] = int(args.seed)

        summary["written"] += 1
        summary["modes"][mode] += 1
        summary["files"].append(
            {
                "slide_id": slide_id,
                "src": str(src_path),
                "dst": str(dst_path),
                "original_n_tiles": n_items,
                "feat_dim": feat_dim,
                "mode": mode,
            }
        )
        print(f"[written] {slide_id} mode={mode} original_tiles={n_items} -> {int(args.target_bag_size)}", flush=True)

    summary_path = out_root.parent / f"fixedbag_{int(args.target_bag_size)}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Fixed bag summary ===", flush=True)
    print("written     =", summary["written"], flush=True)
    print("skipped     =", summary["skipped"], flush=True)
    print("modes       =", summary["modes"], flush=True)
    print("out_root    =", out_root, flush=True)
    print("summary_json=", summary_path, flush=True)


if __name__ == "__main__":
    main()
