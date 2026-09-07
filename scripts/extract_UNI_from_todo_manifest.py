from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import resolve_data_paths
from r2wsp.preprocess.wsi_features import extract_slide_features_to_h5, sanitize_feature_name


def _collect_manifest_slides(manifest: Path) -> list[str]:
    slides = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            slides.append(line)
    return sorted(set(slides))


def _index_svs_files(raw_svs_root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for path in raw_svs_root.rglob("*.svs"):
        if not path.is_file():
            continue
        stem_full = path.stem.lower()
        idx.setdefault(stem_full, path)
        # 无 UUID key (TCGA-XX-XXXX-01A-01-BS1) 给 todo stem 不带 UUID 的情形
        head = stem_full.split(".")[0]
        idx.setdefault(head, path)
        # 兼容 sample_barcode 不含 UUID (LUSC manifest 用无 UUID stem 的常见情况)
        idx.setdefault(stem_full.rsplit(".", 1)[0], path)
    return idx


def _default_feature_root(
    data_root: Path, *, patch_mag: int, patch_size: int, feature_name: str
) -> Path:
    return (
        data_root
        / "wsi_features"
        / f"extracted_mag{int(patch_mag)}x_patch{int(patch_size)}_fp"
        / str(feature_name)
        / "feats_h5"
    ).resolve()


class LocalUniFeatureExtractor:
    model_name = "vit_large_patch16_224"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str,
        batch_size: int = 64,
        amp: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.amp = bool(amp and self.device.type == "cuda")
        ckpt = Path(checkpoint_path).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
        if not isinstance(state, dict):
            raise ValueError(f"unexpected UNI checkpoint payload: {type(state).__name__}")
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    output = self.model(batch)
            else:
                output = self.model(batch)
        if output.ndim != 2:
            raise ValueError(f"expected UNI features to be [B, D], got {tuple(output.shape)}")
        return output.to(dtype=torch.float32)

    def encode_pil_images(self, images: list[Image.Image]) -> "torch.Tensor | list":
        feats: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            batch = torch.stack(
                [self.transform(img) for img in batch_images], dim=0
            ).to(self.device, non_blocking=True)
            output = self._forward(batch).detach().cpu()
            feats.append(output)
        return torch.cat(feats, dim=0).numpy().astype("float32", copy=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract UNI-based WSI feats_h5 for a TCGA cohort using a GDC-official "
        "todo-stem manifest (NO MMP split involvement)."
    )
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument(
        "--todo-manifest",
        required=True,
        help="Path to the stem-level todo list (one stem per line, e.g. UNI_todo_LUSC_FINAL_4AXIOMS_OK.txt)",
    )
    p.add_argument("--cohort", default="", help="Label only, for the summary.")
    p.add_argument("--uni-checkpoint", default="/root/autodl-tmp/R2wsp/data/UNI/pytorch_model.bin")
    p.add_argument("--feature-name", default="uni1024")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patch-mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--target-mpp", type=float, default=0.5)
    p.add_argument("--stride-px", type=int, default=256)
    p.add_argument("--min-tissue-fraction", type=float, default=0.2)
    p.add_argument("--max-tiles", type=int, default=0, help="0 means keep all tissue tiles.")
    p.add_argument("--max-thumbnail-size", type=int, default=2048)
    p.add_argument("--progress-every-batches", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Debug limit on number of slides.")
    p.add_argument(
        "--out-root",
        default=None,
        help="Feature directory to write; typically data/wsi_features/.../feats_h5",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    data_root = paths.data_root.resolve()
    raw_svs_root = paths.raw_svs_root.resolve()
    manifest = Path(args.todo_manifest).resolve()
    cohort = str(args.cohort).upper() or manifest.stem
    feature_name = sanitize_feature_name(args.feature_name)
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

    needed_slides = _collect_manifest_slides(manifest)
    if int(args.limit) > 0:
        needed_slides = needed_slides[: int(args.limit)]
    svs_index = _index_svs_files(raw_svs_root)

    extractor = LocalUniFeatureExtractor(
        checkpoint_path=args.uni_checkpoint,
        device=str(args.device),
        batch_size=int(args.batch_size),
        amp=True,
    )

    summary: dict[str, object] = {
        "cohort": cohort,
        "manifest": str(manifest),
        "raw_svs_root": str(raw_svs_root),
        "out_root": str(out_root),
        "uni_checkpoint": str(Path(args.uni_checkpoint).resolve()),
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
                f"[start] {slide_id} tiles={event['n_tiles']} "
                f"batches={event['total_batches']} partial={event['partial_h5']}",
                flush=True,
            )
        elif kind == "slide_progress":
            print(
                f"[progress] {slide_id} "
                f"batch={event['batch_idx']}/{event['total_batches']} "
                f"tiles={event['tiles_written']}/{event['n_tiles']}",
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
                print(
                    f"[written] {slide_id} tiles={result['n_tiles']} dim={result['feat_dim']}",
                    flush=True,
                )
            else:
                summary["skipped"].append(result)
                print(f"[skipped] {slide_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            summary["failed"].append(
                {"slide_id": str(slide_id), "svs_path": str(svs_path), "error": repr(exc)}
            )
            print(f"[failed] {slide_id} error={exc}", flush=True)

    summary["written_count"] = len(summary["written"])
    summary["skipped_count"] = len(summary["skipped"])
    summary["missing_raw_svs_count"] = len(summary["missing_raw_svs"])
    summary["failed_count"] = len(summary["failed"])

    summary_path = out_root.parent / f"extraction_summary_{cohort}.json"
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
