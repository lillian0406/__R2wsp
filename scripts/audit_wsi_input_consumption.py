from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.build_index import IndexRow


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split_case_ids(split_dir: Path) -> tuple[set[str], dict[str, str]]:
    split_map: dict[str, str] = {}
    for split_name in ("train", "test"):
        split_path = split_dir / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        for row in read_csv_rows(split_path):
            case_id = str(row["case_id"])
            split_map[case_id] = split_name
    return set(split_map.keys()), split_map


def apply_split_filter(rows: list[IndexRow], split_dir: Path | None) -> tuple[list[IndexRow], dict[str, str] | None]:
    if split_dir is None:
        return rows, None
    keep_case_ids, split_map = load_split_case_ids(split_dir)
    filtered = [row for row in rows if row.case_id is not None and str(row.case_id) in keep_case_ids]
    return filtered, split_map


def apply_source_filter(rows: list[IndexRow], source_names: set[str] | None) -> list[IndexRow]:
    if not source_names:
        return rows
    return [row for row in rows if str(row.wsi_feature_source) in source_names]


def dedupe_rows_by_case(rows: list[IndexRow], *, source_name: str | None = None) -> list[IndexRow]:
    grouped: dict[str, list[IndexRow]] = defaultdict(list)
    for row in rows:
        if row.case_id is None:
            continue
        if source_name is not None and str(row.wsi_feature_source) != str(source_name):
            continue
        grouped[str(row.case_id)].append(row)
    chosen: list[IndexRow] = []
    for case_id in sorted(grouped.keys()):
        items = sorted(grouped[case_id], key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))
        chosen.append(items[0])
    return chosen


def infer_tile_count(row: IndexRow) -> int:
    path = Path(row.wsi_feature_path)
    kind = str(row.wsi_feature_kind)
    if kind == "h5_features":
        import h5py

        with h5py.File(path, "r") as f:
            for key in ("feat", "features"):
                if key in f:
                    return int(f[key].shape[0])
        raise ValueError(f"h5 missing feat/features: {path}")
    if kind == "npz_tokens":
        data = np.load(path, allow_pickle=False)
        try:
            if "feat" in data:
                return int(data["feat"].shape[0])
            if "features" in data:
                return int(data["features"].shape[0])
        finally:
            data.close()
        raise ValueError(f"npz missing feat/features: {path}")
    if kind == "pt_features":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if torch.is_tensor(obj):
            return int(obj.shape[0])
        if isinstance(obj, dict):
            feat = obj.get("feat")
            if feat is None:
                feat = obj.get("features")
            if feat is None:
                raise ValueError(f"pt dict missing feat/features: {path}")
            return int(feat.shape[0])
        raise ValueError(f"unsupported pt payload: {type(obj).__name__} @ {path}")
    raise ValueError(f"unsupported feature kind: {kind}")


def consumed_tile_count(raw_tiles: int, max_tiles: int | None) -> int:
    if max_tiles is None or int(max_tiles) <= 0:
        return int(raw_tiles)
    return min(int(raw_tiles), int(max_tiles))


def parse_source_names(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    return set(names) if names else None


def classify_slide_type(slide_id: str) -> str:
    upper = str(slide_id).upper()
    if "DX" in upper:
        return "DX"
    if "TS" in upper:
        return "TS"
    if "BS" in upper:
        return "BS"
    if "MS" in upper:
        return "MS"
    return "OTHER"


def build_case_rows(
    rows: list[IndexRow],
    *,
    max_tiles: int | None,
    split_map: dict[str, str] | None,
) -> list[dict[str, object]]:
    tile_count_cache: dict[str, int] = {}
    rows_by_case_source: dict[tuple[str, str], list[IndexRow]] = defaultdict(list)
    for row in rows:
        if row.case_id is None:
            continue
        key = (str(row.case_id), str(row.wsi_feature_source))
        rows_by_case_source[key].append(row)

    case_rows: list[dict[str, object]] = []
    for (case_id, source_name), group in sorted(rows_by_case_source.items()):
        items = sorted(group, key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))
        split_name = split_map.get(case_id, "all") if split_map is not None else "all"
        raw_tiles_per_slide: list[int] = []
        consumed_tiles_per_slide: list[int] = []
        slide_ids: list[str] = []
        for row in items:
            cache_key = str(row.wsi_feature_path)
            if cache_key not in tile_count_cache:
                tile_count_cache[cache_key] = infer_tile_count(row)
            raw_tiles = int(tile_count_cache[cache_key])
            raw_tiles_per_slide.append(raw_tiles)
            consumed_tiles_per_slide.append(consumed_tile_count(raw_tiles, max_tiles))
            slide_ids.append(str(row.slide_id))

        chosen_single = items[0]
        chosen_raw_tiles = raw_tiles_per_slide[0]
        chosen_consumed_tiles = consumed_tiles_per_slide[0]
        case_rows.append(
            {
                "case_id": case_id,
                "split": split_name,
                "wsi_feature_source": source_name,
                "n_candidate_slides": len(items),
                "candidate_slide_ids": ";".join(slide_ids),
                "single_slide_kept_id": str(chosen_single.slide_id),
                "single_slide_feature_path": str(chosen_single.wsi_feature_path),
                "single_slide_raw_tiles": chosen_raw_tiles,
                "single_slide_consumed_tiles": chosen_consumed_tiles,
                "multi_slide_kept_count": len(items),
                "multi_slide_raw_tiles_total": int(sum(raw_tiles_per_slide)),
                "multi_slide_consumed_tiles_total": int(sum(consumed_tiles_per_slide)),
                "max_tiles_applied_per_slide": "" if max_tiles is None else int(max_tiles),
            }
        )
    return case_rows


def summarize_case_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        groups[str(row["wsi_feature_source"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for source_name in sorted(groups.keys()):
        items = groups[source_name]
        split_counts: dict[str, int] = defaultdict(int)
        total_candidate_slides = 0
        single_raw_tiles = 0
        single_consumed_tiles = 0
        multi_raw_tiles = 0
        multi_consumed_tiles = 0
        multi_kept_slides = 0
        multi_slide_cases = 0
        for row in items:
            split_counts[str(row["split"])] += 1
            n_candidate_slides = int(row["n_candidate_slides"])
            total_candidate_slides += n_candidate_slides
            single_raw_tiles += int(row["single_slide_raw_tiles"])
            single_consumed_tiles += int(row["single_slide_consumed_tiles"])
            multi_raw_tiles += int(row["multi_slide_raw_tiles_total"])
            multi_consumed_tiles += int(row["multi_slide_consumed_tiles_total"])
            multi_kept_slides += int(row["multi_slide_kept_count"])
            if n_candidate_slides > 1:
                multi_slide_cases += 1
        n_cases = len(items)
        summary_rows.append(
            {
                "wsi_feature_source": source_name,
                "n_cases": n_cases,
                "split_case_counts": json.dumps(dict(sorted(split_counts.items())), ensure_ascii=False),
                "candidate_slides_total": total_candidate_slides,
                "candidate_slides_per_case_mean": round(total_candidate_slides / max(n_cases, 1), 4),
                "single_slide_kept_total": n_cases,
                "single_slide_raw_tiles_total": single_raw_tiles,
                "single_slide_consumed_tiles_total": single_consumed_tiles,
                "single_slide_consumed_per_case_mean": round(single_consumed_tiles / max(n_cases, 1), 4),
                "multi_slide_kept_total": multi_kept_slides,
                "multi_slide_cases": multi_slide_cases,
                "multi_slide_raw_tiles_total": multi_raw_tiles,
                "multi_slide_consumed_tiles_total": multi_consumed_tiles,
                "multi_slide_consumed_per_case_mean": round(multi_consumed_tiles / max(n_cases, 1), 4),
                "single_slide_retention_ratio": round(n_cases / max(total_candidate_slides, 1), 6),
                "multi_slide_retention_ratio": round(multi_kept_slides / max(total_candidate_slides, 1), 6),
                "tile_consumption_ratio_single": round(single_consumed_tiles / max(single_raw_tiles, 1), 6),
                "tile_consumption_ratio_multi": round(multi_consumed_tiles / max(multi_raw_tiles, 1), 6),
            }
        )
    return summary_rows


def summarize_slide_types(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        groups[str(row["wsi_feature_source"])].append(row)

    out_rows: list[dict[str, object]] = []
    for source_name in sorted(groups.keys()):
        items = groups[source_name]
        kept_counts: dict[str, int] = defaultdict(int)
        dx_candidate_cases = 0
        dx_candidate_but_not_kept = 0
        multi_case_count = 0
        for row in items:
            kept_type = classify_slide_type(str(row["single_slide_kept_id"]))
            kept_counts[kept_type] += 1
            if int(row["n_candidate_slides"]) > 1:
                multi_case_count += 1
            candidate_types = {classify_slide_type(x) for x in str(row["candidate_slide_ids"]).split(";") if x}
            if "DX" in candidate_types:
                dx_candidate_cases += 1
                if kept_type != "DX":
                    dx_candidate_but_not_kept += 1
        n_cases = len(items)
        out_rows.append(
            {
                "wsi_feature_source": source_name,
                "n_cases": n_cases,
                "multi_case_count": multi_case_count,
                "kept_dx_count": kept_counts.get("DX", 0),
                "kept_ts_count": kept_counts.get("TS", 0),
                "kept_bs_count": kept_counts.get("BS", 0),
                "kept_ms_count": kept_counts.get("MS", 0),
                "kept_other_count": kept_counts.get("OTHER", 0),
                "dx_candidate_cases": dx_candidate_cases,
                "dx_candidate_but_not_kept": dx_candidate_but_not_kept,
                "dx_candidate_but_not_kept_ratio": round(dx_candidate_but_not_kept / max(dx_candidate_cases, 1), 6),
            }
        )
    return out_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print("=== WSI Input Audit Summary ===", flush=True)
    for row in summary_rows:
        print(
            (
                f"[{row['wsi_feature_source']}] "
                f"cases={row['n_cases']} "
                f"candidate_slides_total={row['candidate_slides_total']} "
                f"single_keep={row['single_slide_kept_total']} "
                f"multi_keep={row['multi_slide_kept_total']} "
                f"single_tiles={row['single_slide_consumed_tiles_total']} "
                f"multi_tiles={row['multi_slide_consumed_tiles_total']} "
                f"single_ratio={row['tile_consumption_ratio_single']} "
                f"multi_ratio={row['tile_consumption_ratio_multi']}"
            ),
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Audit how many raw WSI tiles/slides are actually retained by current direct-input entry assumptions."
    )
    p.add_argument("--project-root", default=str(_ROOT))
    p.add_argument("--data-root", default=None)
    p.add_argument("--wsi-feature-source", default=None, help="Comma-separated source names. Default audits all sources.")
    p.add_argument("--split-dir", default=None, help="Optional official split dir to mirror a concrete training run.")
    p.add_argument("--max-tiles", type=int, default=256, help="Per-slide max_tiles used by current loader semantics.")
    p.add_argument("--limit-cases", type=int, default=0, help="Optional limit after filtering, for quick debugging.")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = resolve_data_paths(project_root=project_root, data_root=args.data_root)
    paths.validate()

    inventory = scan_all_assets(paths)
    all_rows = build_index(inventory)
    source_names = parse_source_names(args.wsi_feature_source)
    rows = apply_source_filter(all_rows, source_names)
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    rows, split_map = apply_split_filter(rows, split_dir)
    rows = [row for row in rows if row.case_id is not None]
    if int(args.limit_cases) > 0:
        seen: set[str] = set()
        limited: list[IndexRow] = []
        for row in rows:
            case_id = str(row.case_id)
            if case_id not in seen and len(seen) >= int(args.limit_cases):
                continue
            seen.add(case_id)
            limited.append(row)
        rows = limited
        if split_map is not None:
            split_map = {k: v for k, v in split_map.items() if k in seen}

    case_rows = build_case_rows(rows, max_tiles=args.max_tiles, split_map=split_map)
    summary_rows = summarize_case_rows(case_rows)
    print_summary(summary_rows)

    split_tag = Path(args.split_dir).name if args.split_dir else "all_cases"
    source_tag = "all_sources" if not source_names else "_".join(sorted(source_names))
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else project_root / "outputs" / "wsi_input_audit" / f"{source_tag}__{split_tag}__maxtiles{int(args.max_tiles)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "source_summary.csv"
    case_path = out_dir / "case_details.csv"
    slide_type_summary_path = out_dir / "slide_type_summary.csv"
    meta_path = out_dir / "meta.json"
    slide_type_rows = summarize_slide_types(case_rows)
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()) if summary_rows else ["wsi_feature_source"])
    write_csv(case_path, case_rows, list(case_rows[0].keys()) if case_rows else ["case_id"])
    write_csv(
        slide_type_summary_path,
        slide_type_rows,
        list(slide_type_rows[0].keys()) if slide_type_rows else ["wsi_feature_source"],
    )
    meta = {
        "project_root": str(project_root),
        "data_root": str(paths.data_root),
        "split_dir": str(split_dir) if split_dir is not None else None,
        "wsi_feature_source": sorted(source_names) if source_names else None,
        "max_tiles": int(args.max_tiles),
        "n_rows_after_filter": len(rows),
        "n_case_rows": len(case_rows),
        "summary_csv": str(summary_path),
        "case_details_csv": str(case_path),
        "slide_type_summary_csv": str(slide_type_summary_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_csv={summary_path}", flush=True)
    print(f"case_details_csv={case_path}", flush=True)
    print(f"slide_type_summary_csv={slide_type_summary_path}", flush=True)
    print(f"meta_json={meta_path}", flush=True)


if __name__ == "__main__":
    main()
