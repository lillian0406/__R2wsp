from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.build_index import IndexRow

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_split_case_ids(split_dir: Path) -> tuple[set[str], dict[str, str]]:
    split_map: dict[str, str] = {}
    for split_name in ("train", "test"):
        split_path = split_dir / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        for row in read_csv_rows(split_path):
            split_map[str(row["case_id"])] = split_name
    return set(split_map.keys()), split_map


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


def slide_priority(slide_id: str) -> int:
    cls = classify_slide_type(slide_id)
    if cls == "DX":
        return 0
    if cls == "TS":
        return 1
    if cls == "BS":
        return 2
    if cls == "MS":
        return 3
    return 4


def has_uuid_token(slide_id: str) -> bool:
    return UUID_RE.search(str(slide_id)) is not None


def selection_rank(row: IndexRow) -> tuple[int, int, int, int, str, str]:
    slide_id = str(row.slide_id)
    path_str = str(row.wsi_feature_path)
    return (
        slide_priority(slide_id),
        0 if row.svs_path is not None else 1,
        0 if has_uuid_token(slide_id) else 1,
        -len(slide_id),
        slide_id.lower(),
        path_str.lower(),
    )


def select_single_slide_rows(
    rows: list[IndexRow],
    *,
    split_map: dict[str, str] | None,
) -> tuple[list[IndexRow], list[dict[str, object]]]:
    groups: dict[str, list[IndexRow]] = defaultdict(list)
    for row in rows:
        if row.case_id is None:
            continue
        groups[str(row.case_id)].append(row)

    selected_rows: list[IndexRow] = []
    selection_rows: list[dict[str, object]] = []
    for case_id in sorted(groups.keys()):
        items = sorted(groups[case_id], key=selection_rank)
        chosen = items[0]
        candidate_ids = [str(x.slide_id) for x in items]
        candidate_types = [classify_slide_type(str(x.slide_id)) for x in items]
        selected_rows.append(chosen)
        selection_rows.append(
            {
                "case_id": case_id,
                "split": split_map.get(case_id, "all") if split_map is not None else "all",
                "source_name": str(chosen.wsi_feature_source),
                "selected_slide_id": str(chosen.slide_id),
                "selected_slide_type": classify_slide_type(str(chosen.slide_id)),
                "selected_feature_path": str(chosen.wsi_feature_path),
                "n_candidate_slides": len(items),
                "candidate_slide_ids": ";".join(candidate_ids),
                "candidate_slide_types": ";".join(candidate_types),
                "has_dx_candidate": "DX" in set(candidate_types),
                "selected_has_uuid": has_uuid_token(str(chosen.slide_id)),
            }
        )
    return selected_rows, selection_rows


def infer_output_root(paths, selected_rows: list[IndexRow], output_source_name: str) -> Path:
    if not selected_rows:
        raise ValueError("selected_rows is empty")
    kind = str(selected_rows[0].wsi_feature_kind)
    if kind == "npz_tokens":
        return (paths.token_root / output_source_name).resolve()
    if kind == "h5_features":
        return (paths.data_root / "wsi_features" / "reselected_single_slide" / output_source_name / "feats_h5").resolve()
    if kind == "pt_features":
        return (paths.data_root / "wsi_features" / "reselected_single_slide" / output_source_name / "feats_pt").resolve()
    raise ValueError(f"unsupported feature kind: {kind}")


def materialize_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(f"unsupported link mode: {mode}")


def rewrite_split_csvs(split_dir: Path, out_split_root: Path, selected_by_case: dict[str, dict[str, object]]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for split_name in ("train", "test"):
        src_path = split_dir / f"{split_name}.csv"
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        rows = read_csv_rows(src_path)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["case_id"])].append(row)

        out_rows: list[dict[str, str]] = []
        dropped_cases = 0
        for case_id in sorted(grouped.keys()):
            selected = selected_by_case.get(case_id)
            if selected is None:
                dropped_cases += 1
                continue
            row = dict(grouped[case_id][0])
            row["slide_id"] = str(selected["selected_slide_id"])
            out_rows.append(row)

        out_path = out_split_root / f"{split_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_rows:
            fieldnames = list(out_rows[0].keys())
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in out_rows:
                    writer.writerow(row)
        else:
            fieldnames = list(rows[0].keys()) if rows else ["case_id", "slide_id"]
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        stats[split_name] = {
            "input_rows": len(rows),
            "output_rows": len(out_rows),
            "dropped_cases": dropped_cases,
            "path": str(out_path),
        }
    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a reselected single-slide feature source with DX > TS > BS > MS > OTHER priority without touching main training files."
    )
    p.add_argument("--project-root", default=str(_ROOT))
    p.add_argument("--data-root", default=None)
    p.add_argument("--wsi-feature-source", required=True)
    p.add_argument("--output-source-name", default=None)
    p.add_argument("--split-dir", default=None, help="Optional split dir. When provided, only cases in this split are materialized.")
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--out-root", default=None, help="Optional explicit output feature directory.")
    p.add_argument("--manifest-dir", default=None, help="Optional explicit manifest directory.")
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = resolve_data_paths(project_root=project_root, data_root=args.data_root)
    paths.validate()

    inventory = scan_all_assets(paths)
    all_rows = build_index(inventory)
    source_name = str(args.wsi_feature_source)
    rows = [row for row in all_rows if str(row.wsi_feature_source) == source_name and row.case_id is not None]
    if not rows:
        raise ValueError(f"no rows found for source: {source_name}")

    split_map: dict[str, str] | None = None
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    if split_dir is not None:
        keep_case_ids, split_map = load_split_case_ids(split_dir)
        rows = [row for row in rows if str(row.case_id) in keep_case_ids]
        if not rows:
            raise ValueError(f"no rows remain for source={source_name} after split filter: {split_dir}")

    selected_rows, selection_rows = select_single_slide_rows(rows, split_map=split_map)
    output_source_name = str(args.output_source_name or f"{source_name}_single_dx_ts_bs")
    feature_out_root = Path(args.out_root).resolve() if args.out_root else infer_output_root(paths, selected_rows, output_source_name)
    manifest_dir = (
        Path(args.manifest_dir).resolve()
        if args.manifest_dir
        else project_root / "outputs" / "single_slide_reselection" / output_source_name
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    materialized_rows: list[dict[str, object]] = []
    for row, selected in zip(selected_rows, selection_rows):
        src = Path(row.wsi_feature_path).resolve()
        dst = feature_out_root / src.name
        materialized_as = materialize_file(src, dst, str(args.link_mode))
        materialized_rows.append(
            {
                **selected,
                "materialized_path": str(dst),
                "materialized_as": materialized_as,
            }
        )

    selected_by_case = {str(row["case_id"]): row for row in materialized_rows}
    split_stats = None
    rewritten_split_root = None
    if split_dir is not None:
        rewritten_split_root = (manifest_dir / "rewritten_split").resolve()
        split_stats = rewrite_split_csvs(split_dir, rewritten_split_root, selected_by_case)

    kept_type_counts: dict[str, int] = defaultdict(int)
    dx_candidate_cases = 0
    dx_candidate_but_not_kept = 0
    for row in materialized_rows:
        kept_type_counts[str(row["selected_slide_type"])] += 1
        has_dx_candidate = bool(row["has_dx_candidate"])
        if has_dx_candidate:
            dx_candidate_cases += 1
            if str(row["selected_slide_type"]) != "DX":
                dx_candidate_but_not_kept += 1

    selection_csv = manifest_dir / "selection_manifest.csv"
    summary_json = manifest_dir / "summary.json"
    write_csv(
        selection_csv,
        materialized_rows,
        list(materialized_rows[0].keys()) if materialized_rows else ["case_id"],
    )
    summary = {
        "source_name": source_name,
        "output_source_name": output_source_name,
        "feature_out_root": str(feature_out_root),
        "link_mode": str(args.link_mode),
        "selected_cases": len(materialized_rows),
        "selected_type_counts": dict(sorted(kept_type_counts.items())),
        "dx_candidate_cases": dx_candidate_cases,
        "dx_candidate_but_not_kept": dx_candidate_but_not_kept,
        "dx_candidate_but_not_kept_ratio": round(dx_candidate_but_not_kept / max(dx_candidate_cases, 1), 6),
        "selection_manifest_csv": str(selection_csv),
        "rewritten_split_root": str(rewritten_split_root) if rewritten_split_root is not None else None,
        "split_stats": split_stats,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Single-Slide Reselection Summary ===", flush=True)
    print(f"source_name={source_name}", flush=True)
    print(f"output_source_name={output_source_name}", flush=True)
    print(f"feature_out_root={feature_out_root}", flush=True)
    print(f"selected_cases={len(materialized_rows)}", flush=True)
    print(f"selected_type_counts={dict(sorted(kept_type_counts.items()))}", flush=True)
    print(f"dx_candidate_but_not_kept={dx_candidate_but_not_kept}", flush=True)
    print(f"selection_manifest_csv={selection_csv}", flush=True)
    if rewritten_split_root is not None:
        print(f"rewritten_split_root={rewritten_split_root}", flush=True)
    print(f"summary_json={summary_json}", flush=True)


if __name__ == "__main__":
    main()
