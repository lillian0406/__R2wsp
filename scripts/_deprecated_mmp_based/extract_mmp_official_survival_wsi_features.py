from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import resolve_data_paths
from r2wsp.preprocess.wsi_features import (
    TimmFeatureExtractor,
    extract_slide_features_to_h5,
    sanitize_feature_name,
)


def _collect_needed_slides(split_root: Path, cohort: str, n_folds: int) -> list[str]:
    import pandas as pd

    needed: set[str] = set()
    for k in range(int(n_folds)):
        fold_dir = split_root / f"TCGA_{cohort}_overall_survival_k={k}"
        for split_name in ("train.csv", "test.csv"):
            split_path = fold_dir / split_name
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            df = pd.read_csv(split_path)
            needed.update(df["slide_id"].astype(str).tolist())
    return sorted(needed)


def _index_svs_files(raw_svs_root: Path) -> dict[str, Path]:
    return {path.stem.lower(): path for path in raw_svs_root.rglob("*.svs") if path.is_file()}


def _default_feature_root(data_root: Path, *, patch_mag: int, patch_size: int, feature_name: str) -> Path:
    return (
        data_root
        / "wsi_features"
        / f"extracted_mag{int(patch_mag)}x_patch{int(patch_size)}_fp"
        / str(feature_name)
        / "feats_h5"
    ).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="Extract MMP-style WSI feats_h5 for official survival split slides from raw SVS. Use --pretrained 0 with --checkpoint-path for offline cloud runs.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--mmp-root", required=True)
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--model-name", default="vit_large_patch14_dinov2.lvd142m")
    p.add_argument("--feature-name", default=None)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--checkpoint-path", default=None, help="Local timm checkpoint path for offline extraction.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch-mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--target-mpp", type=float, default=0.5)
    p.add_argument("--stride-px", type=int, default=256)
    p.add_argument("--min-tissue-fraction", type=float, default=0.2)
    p.add_argument("--max-tiles", type=int, default=0, help="0 means keep all selected tissue tiles.")
    p.add_argument("--max-thumbnail-size", type=int, default=2048)
    p.add_argument("--progress-every-batches", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Debug limit on number of slides to process.")
    p.add_argument("--out-root", default=None, help="Feature directory to write, typically .../feats_h5.")
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    data_root = paths.data_root.resolve()
    raw_svs_root = paths.raw_svs_root.resolve()
    mmp_root = Path(args.mmp_root).resolve()
    split_root = (mmp_root / "src" / "splits" / "survival").resolve()
    cohort = str(args.cohort).upper()
    feature_name = sanitize_feature_name(args.feature_name or args.model_name)
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else _default_feature_root(
            data_root,
            patch_mag=int(args.patch_mag),
            patch_size=int(args.patch_size),
            feature_name=feature_name,
        )
    )
    out_root.mkdir(parents=True, exist_ok=True)

    needed_slides = _collect_needed_slides(split_root, cohort=cohort, n_folds=int(args.n_folds))
    if int(args.limit) > 0:
        needed_slides = needed_slides[: int(args.limit)]
    svs_index = _index_svs_files(raw_svs_root)

    extractor = TimmFeatureExtractor(
        model_name=str(args.model_name),
        device=str(args.device),
        pretrained=bool(int(args.pretrained)),
        checkpoint_path=args.checkpoint_path,
        batch_size=int(args.batch_size),
        amp=True,
    )

    summary: dict[str, object] = {
        "cohort": cohort,
        "mmp_root": str(mmp_root),
        "raw_svs_root": str(raw_svs_root),
        "out_root": str(out_root),
        "model_name": str(args.model_name),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()) if args.checkpoint_path else None,
        "feature_name": feature_name,
        "target_mpp": float(args.target_mpp),
        "patch_size": int(args.patch_size),
        "needed_slides": int(len(needed_slides)),
        "written": [],
        "skipped": [],
        "missing_raw_svs": [],
        "failed": [],
    }

    def log_event(event: dict[str, object]) -> None:
        kind = str(event.get("event"))
        slide_id = str(event.get("slide_id"))
        if kind == "slide_start":
            print(
                f"[start] {slide_id} tiles={event['n_tiles']} batches={event['total_batches']} partial={event['partial_h5']}",
                flush=True,
            )
        elif kind == "slide_progress":
            print(
                f"[progress] {slide_id} batch={event['batch_idx']}/{event['total_batches']} tiles={event['tiles_written']}/{event['n_tiles']}",
                flush=True,
            )

    for slide_id in needed_slides:
        svs_path = svs_index.get(str(slide_id).lower())
        if svs_path is None:
            summary["missing_raw_svs"].append(str(slide_id))
            print(f"[missing] {slide_id}")
            continue

        output_h5 = out_root / f"{slide_id}.h5"
        try:
            result = extract_slide_features_to_h5(
                wsi_path=svs_path,
                output_h5=output_h5,
                extractor=extractor,
                tile_px=int(args.patch_size),
                target_mpp=float(args.target_mpp),
                stride_px=int(args.stride_px),
                max_thumbnail_size=int(args.max_thumbnail_size),
                min_tissue_fraction=float(args.min_tissue_fraction),
                max_tiles=int(args.max_tiles) if int(args.max_tiles) > 0 else None,
                edge=False,
                overwrite=bool(args.overwrite),
                progress_every_batches=int(args.progress_every_batches),
                progress_callback=log_event,
            )
            status = str(result.get("status"))
            if status == "written":
                summary["written"].append(result)
                print(f"[written] {slide_id} tiles={result['n_tiles']} dim={result['feat_dim']}", flush=True)
            else:
                summary["skipped"].append(result)
                print(f"[skipped] {slide_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            summary["failed"].append({"slide_id": str(slide_id), "svs_path": str(svs_path), "error": repr(exc)})
            print(f"[failed] {slide_id} error={exc}", flush=True)

    summary["written_count"] = len(summary["written"])
    summary["skipped_count"] = len(summary["skipped"])
    summary["missing_raw_svs_count"] = len(summary["missing_raw_svs"])
    summary["failed_count"] = len(summary["failed"])

    summary_path = out_root.parent / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Extraction summary ===", flush=True)
    print("written_count        =", summary["written_count"], flush=True)
    print("skipped_count        =", summary["skipped_count"], flush=True)
    print("missing_raw_svs_count=", summary["missing_raw_svs_count"], flush=True)
    print("failed_count         =", summary["failed_count"], flush=True)
    print("out_root             =", out_root, flush=True)
    print("summary_json         =", summary_path, flush=True)


if __name__ == "__main__":
    main()
