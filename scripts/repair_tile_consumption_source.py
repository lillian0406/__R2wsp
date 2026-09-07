from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.build_index import IndexRow


def read_h5_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object], str, str]:
    with h5py.File(path, "r") as f:
        feat = None
        feat_key = ""
        for key in ("feat", "features"):
            if key in f:
                feat = np.asarray(f[key], dtype=np.float32)
                feat_key = key
                break
        if feat is None:
            raise ValueError(f"h5 missing feat/features: {path}")
        xy = None
        xy_key = ""
        for key in ("xy", "coords"):
            if key in f:
                xy = np.asarray(f[key])
                xy_key = key
                break
        if xy is None:
            xy = np.zeros((feat.shape[0], 2), dtype=np.float32)
            xy_key = "coords"
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    return feat, np.asarray(xy), attrs, feat_key, xy_key


def spaced_positions(n_items: int, target_tiles: int) -> np.ndarray:
    if target_tiles <= 0 or n_items <= target_tiles:
        return np.arange(n_items, dtype=np.int64)
    pos = np.floor((np.arange(target_tiles, dtype=np.float64) + 0.5) * float(n_items) / float(target_tiles)).astype(np.int64)
    pos = np.clip(pos, 0, n_items - 1)
    _, first_idx = np.unique(pos, return_index=True)
    if len(first_idx) != len(pos):
        pos = np.floor(np.arange(target_tiles, dtype=np.float64) * float(n_items) / float(target_tiles)).astype(np.int64)
        pos = np.clip(pos, 0, n_items - 1)
    return pos.astype(np.int64)


def select_tile_indices(feat: np.ndarray, xy: np.ndarray, target_tiles: int) -> tuple[np.ndarray, str]:
    n_items = int(feat.shape[0])
    if target_tiles <= 0 or n_items <= target_tiles:
        return np.arange(n_items, dtype=np.int64), "keep_all"
    if xy.ndim == 2 and xy.shape == (n_items, 2) and np.unique(xy, axis=0).shape[0] > 1:
        order = np.lexsort((np.asarray(xy[:, 0]), np.asarray(xy[:, 1])))
        mode = "spatial_stride"
    else:
        order = np.arange(n_items, dtype=np.int64)
        mode = "index_stride"
    selected_positions = spaced_positions(n_items, target_tiles)
    selected_ids = order[selected_positions]
    return np.asarray(selected_ids, dtype=np.int64), mode


def infer_out_root(first_row: IndexRow, output_source_name: str) -> Path:
    path = Path(first_row.wsi_feature_path).resolve()
    feats_dir = path.parent
    if feats_dir.name != "feats_h5":
        raise ValueError(f"expected h5 source under feats_h5, got: {feats_dir}")
    return (feats_dir.parent.parent / output_source_name / "feats_h5").resolve()


def write_h5(
    dst_path: Path,
    *,
    feat: np.ndarray,
    xy: np.ndarray,
    attrs: dict[str, object],
    feat_key: str,
    xy_key: str,
    src_path: Path,
    target_tiles: int,
    selection_mode: str,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_path, "w") as f:
        f.create_dataset(feat_key, data=np.asarray(feat, dtype=np.float32), dtype=np.float32, compression="gzip")
        xy_dtype = np.asarray(xy).dtype if np.asarray(xy).dtype.kind in {"i", "u", "f"} else np.float32
        f.create_dataset(xy_key, data=np.asarray(xy, dtype=xy_dtype), dtype=xy_dtype, compression="gzip")
        for key, value in attrs.items():
            f.attrs[key] = value
        f.attrs["tile_fix_source_path"] = str(src_path)
        f.attrs["tile_fix_target_tiles"] = int(target_tiles)
        f.attrs["tile_fix_selection_mode"] = str(selection_mode)
        f.attrs["tile_fix_version"] = "v1"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a tile-consumption-fixed h5 source so current max_tiles consumes a slide-wide distributed subset instead of the first prefix."
    )
    p.add_argument("--project-root", default=str(_ROOT))
    p.add_argument("--data-root", default=None)
    p.add_argument("--wsi-feature-source", required=True)
    p.add_argument("--target-tiles", type=int, default=256)
    p.add_argument("--output-source-name", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = resolve_data_paths(project_root=project_root, data_root=args.data_root)
    paths.validate()

    inventory = scan_all_assets(paths)
    all_rows = build_index(inventory)
    source_name = str(args.wsi_feature_source)
    rows = [row for row in all_rows if str(row.wsi_feature_source) == source_name]
    if not rows:
        raise ValueError(f"no rows found for source: {source_name}")
    if any(str(r.wsi_feature_kind) != "h5_features" for r in rows):
        raise ValueError(f"tile repair currently supports h5_features only: {source_name}")

    unique_rows_by_path: dict[str, IndexRow] = {}
    for row in rows:
        unique_rows_by_path[str(Path(row.wsi_feature_path).resolve())] = row
    unique_rows = [unique_rows_by_path[k] for k in sorted(unique_rows_by_path.keys())]
    if int(args.limit) > 0:
        unique_rows = unique_rows[: int(args.limit)]
    if not unique_rows:
        raise ValueError("no unique source files to process")

    output_source_name = str(args.output_source_name or f"{source_name}_tilefix{int(args.target_tiles)}_spatial")
    out_root = Path(args.out_root).resolve() if args.out_root else infer_out_root(unique_rows[0], output_source_name)
    manifest_dir = (
        Path(args.manifest_dir).resolve()
        if args.manifest_dir
        else project_root / "outputs" / "tile_consumption_repair" / output_source_name
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    mode_counts: dict[str, int] = defaultdict(int)
    for row in unique_rows:
        src_path = Path(row.wsi_feature_path).resolve()
        dst_path = out_root / src_path.name
        if dst_path.exists() and not bool(args.overwrite):
            continue
        feat, xy, attrs, feat_key, xy_key = read_h5_arrays(src_path)
        selected_ids, mode = select_tile_indices(feat, xy, int(args.target_tiles))
        fixed_feat = feat[selected_ids]
        fixed_xy = xy[selected_ids]
        write_h5(
            dst_path,
            feat=fixed_feat,
            xy=fixed_xy,
            attrs=attrs,
            feat_key=feat_key,
            xy_key=xy_key,
            src_path=src_path,
            target_tiles=int(args.target_tiles),
            selection_mode=mode,
        )
        mode_counts[mode] += 1
        manifest_rows.append(
            {
                "slide_id": str(row.slide_id),
                "case_id": str(row.case_id) if row.case_id is not None else "",
                "source_name": source_name,
                "output_source_name": output_source_name,
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "original_n_tiles": int(feat.shape[0]),
                "written_n_tiles": int(fixed_feat.shape[0]),
                "selection_mode": mode,
            }
        )

    manifest_csv = manifest_dir / "manifest.csv"
    summary_json = manifest_dir / "summary.json"
    write_csv(manifest_csv, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["slide_id"])
    summary = {
        "source_name": source_name,
        "output_source_name": output_source_name,
        "target_tiles": int(args.target_tiles),
        "out_root": str(out_root),
        "processed_files": len(manifest_rows),
        "mode_counts": dict(sorted(mode_counts.items())),
        "manifest_csv": str(manifest_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Tile Consumption Repair Summary ===", flush=True)
    print(f"source_name={source_name}", flush=True)
    print(f"output_source_name={output_source_name}", flush=True)
    print(f"target_tiles={int(args.target_tiles)}", flush=True)
    print(f"out_root={out_root}", flush=True)
    print(f"processed_files={len(manifest_rows)}", flush=True)
    print(f"mode_counts={dict(sorted(mode_counts.items()))}", flush=True)
    print(f"manifest_csv={manifest_csv}", flush=True)
    print(f"summary_json={summary_json}", flush=True)


if __name__ == "__main__":
    main()
