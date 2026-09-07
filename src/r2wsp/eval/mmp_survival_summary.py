from __future__ import annotations

import csv
import json
import math
import pickle
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2wsp.eval.survival_stats import logrank_p_value_from_risk


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
_CINDEX_LINE_RE = re.compile(r"(?:c[\s\-_]*index|cindex)\D*([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)", re.IGNORECASE)
_LOGRANK_LINE_RE = re.compile(
    r"(?:log[\s\-_]*rank(?:[\s\-_]*p(?:[\s\-_]*value)?)?|p[\s\-_]*value)\D*([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

_CINDEX_KEYS = [
    "test_c_index",
    "test_cindex",
    "c_index",
    "cindex",
    "val_c_index",
    "val_cindex",
    "valid_c_index",
    "valid_cindex",
]
_LOGRANK_KEYS = [
    "logrank_p_value",
    "log_rank_p_value",
    "logrank_p",
    "log_rank_p",
    "p_value",
]


@dataclass
class FoldMetrics:
    fold: int
    c_index: float | None
    logrank_p: float | None
    metric_source: str | None
    dump_path: str | None
    summary_path: str | None
    log_path: str | None
    notes: list[str]


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isfinite(x):
            return x
        return None
    s = str(value).strip()
    if not s:
        return None
    m = _FLOAT_RE.search(s)
    if m is None:
        return None
    try:
        x = float(m.group(0))
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _iter_flat_items(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_iter_flat_items(value, child))
        return items
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]"
            items.extend(_iter_flat_items(value, child))
        return items
    items.append((prefix, obj))
    return items


def _score_key(name: str, preferred: list[str], *, metric: str) -> tuple[int, int, str]:
    norm = _normalize_key(name)
    try:
        idx = preferred.index(norm)
        return (0, idx, norm)
    except ValueError:
        pass
    if metric == "c_index" and ("c_index" in norm or "cindex" in norm):
        return (1, 0, norm)
    if metric == "logrank_p" and "logrank" in norm and "p" in norm:
        return (1, 0, norm)
    return (9, 9, norm)


def _pick_metric(flat_items: list[tuple[str, Any]], *, metric: str) -> tuple[float | None, str | None]:
    preferred = _CINDEX_KEYS if metric == "c_index" else _LOGRANK_KEYS
    candidates: list[tuple[tuple[int, int, str], float, str]] = []
    for key, value in flat_items:
        score = _score_key(key, preferred, metric=metric)
        if score[0] >= 9:
            continue
        number = _to_float(value)
        if number is None:
            continue
        candidates.append((score, number, key))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, number, key = candidates[0]
    return number, key


def _read_json_metrics(path: Path) -> tuple[float | None, float | None, list[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    flat_items = _iter_flat_items(obj)
    c_index, c_key = _pick_metric(flat_items, metric="c_index")
    logrank_p, p_key = _pick_metric(flat_items, metric="logrank_p")
    notes: list[str] = []
    if c_key:
        notes.append(f"json:c_index={c_key}")
    if p_key:
        notes.append(f"json:logrank_p={p_key}")
    return c_index, logrank_p, notes


def _read_csv_metrics(path: Path) -> tuple[float | None, float | None, list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, ["csv:empty"]
    ranked_rows: list[tuple[tuple[int, int, str], float, float | None, str, str | None]] = []
    for row_idx, row in enumerate(rows):
        flat_items = list(row.items())
        c_index, c_key = _pick_metric(flat_items, metric="c_index")
        logrank_p, p_key = _pick_metric(flat_items, metric="logrank_p")
        if c_index is None and logrank_p is None:
            continue
        score = _score_key(c_key or "", _CINDEX_KEYS, metric="c_index") if c_key else (8, 8, "")
        ranked_rows.append((score, c_index if c_index is not None else -1.0, logrank_p, c_key or f"row[{row_idx}]", p_key))
    if not ranked_rows:
        return None, None, ["csv:no_metric_row"]
    ranked_rows.sort(key=lambda x: (x[0], -x[1]))
    _, c_index, logrank_p, c_key, p_key = ranked_rows[0]
    notes = [f"csv:c_index={c_key}"]
    if p_key:
        notes.append(f"csv:logrank_p={p_key}")
    return (None if c_index < 0 else c_index), logrank_p, notes


def _read_pickle_metrics(path: Path) -> tuple[float | None, float | None, list[str]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    flat_items = _iter_flat_items(obj)
    c_index, c_key = _pick_metric(flat_items, metric="c_index")
    logrank_p, p_key = _pick_metric(flat_items, metric="logrank_p")
    notes: list[str] = []
    if c_key:
        notes.append(f"pickle:c_index={c_key}")
    if p_key:
        notes.append(f"pickle:logrank_p={p_key}")
    return c_index, logrank_p, notes


def _extract_split_arrays(obj: Any, split_name: str) -> tuple[Any, Any, Any] | None:
    if not isinstance(obj, dict):
        return None
    split = obj.get(split_name)
    if not isinstance(split, dict):
        return None
    event_times = split.get("event_times")
    censorships = split.get("censorships")
    risk_scores = split.get("risk_scores")
    if event_times is None or censorships is None or risk_scores is None:
        return None
    return event_times, censorships, risk_scores


def _compute_logrank_from_dump(path: Path) -> tuple[float | None, list[str]]:
    with path.open("rb") as f:
        obj = pickle.load(f)

    for split_name in ["test", "val", "train"]:
        arrays = _extract_split_arrays(obj, split_name)
        if arrays is None:
            continue
        result = logrank_p_value_from_risk(*arrays)
        notes = [f"dump_logrank_split={split_name}", *result.notes]
        if result.threshold is not None:
            notes.append(f"dump_logrank_threshold={result.threshold:.6g}")
        notes.append(f"dump_logrank_groups=high:{result.group_high},low:{result.group_low}")
        return result.p_value, notes
    return None, ["dump_logrank_missing_arrays"]


def _read_log_metrics(path: Path) -> tuple[float | None, float | None, list[str]]:
    c_index: float | None = None
    logrank_p: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        c_match = _CINDEX_LINE_RE.search(line)
        if c_match is not None:
            c_index = _to_float(c_match.group(1))
        p_match = _LOGRANK_LINE_RE.search(line)
        if p_match is not None and "log" in line.lower():
            logrank_p = _to_float(p_match.group(1))
    notes: list[str] = []
    if c_index is not None:
        notes.append("log:c_index")
    if logrank_p is not None:
        notes.append("log:logrank_p")
    return c_index, logrank_p, notes


def _find_metric_artifacts(fold_dir: Path) -> dict[str, Path | None]:
    summary_candidates = [
        *sorted(fold_dir.glob("**/summary_fixed.csv")),
        *sorted(fold_dir.glob("**/summary.csv")),
        *sorted(fold_dir.glob("**/summary*.json")),
    ]
    dump_candidates = sorted(fold_dir.glob("**/all_dumps.h5"))
    log_candidates = sorted(fold_dir.glob("**/train.log"))
    return {
        "summary": summary_candidates[-1] if summary_candidates else None,
        "dump": dump_candidates[-1] if dump_candidates else None,
        "log": log_candidates[-1] if log_candidates else None,
    }


def collect_fold_metrics(fold_dir: Path, *, fold: int) -> FoldMetrics:
    artifacts = _find_metric_artifacts(fold_dir)
    notes: list[str] = []
    c_index: float | None = None
    logrank_p: float | None = None
    metric_source: str | None = None

    summary_path = artifacts["summary"]
    if summary_path is not None:
        try:
            if summary_path.suffix.lower() == ".csv":
                c_index, logrank_p, sub_notes = _read_csv_metrics(summary_path)
            else:
                c_index, logrank_p, sub_notes = _read_json_metrics(summary_path)
            if c_index is not None or logrank_p is not None:
                metric_source = str(summary_path)
            notes.extend(sub_notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"summary_error={exc}")

    dump_path = artifacts["dump"]
    if (c_index is None or logrank_p is None) and dump_path is not None:
        try:
            dump_c, dump_p, sub_notes = _read_pickle_metrics(dump_path)
            if c_index is None:
                c_index = dump_c
            if logrank_p is None:
                logrank_p = dump_p
            derived_p: float | None = None
            if logrank_p is None:
                derived_p, derived_notes = _compute_logrank_from_dump(dump_path)
                if derived_p is not None:
                    logrank_p = derived_p
                notes.extend(derived_notes)
            if (dump_c is not None or dump_p is not None or derived_p is not None) and metric_source is None:
                metric_source = str(dump_path)
            notes.extend(sub_notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"dump_error={exc}")

    log_path = artifacts["log"]
    if (c_index is None or logrank_p is None) and log_path is not None:
        try:
            log_c, log_p, sub_notes = _read_log_metrics(log_path)
            if c_index is None:
                c_index = log_c
            if logrank_p is None:
                logrank_p = log_p
            if (log_c is not None or log_p is not None) and metric_source is None:
                metric_source = str(log_path)
            notes.extend(sub_notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"log_error={exc}")

    if c_index is None:
        notes.append("missing_c_index")

    return FoldMetrics(
        fold=int(fold),
        c_index=c_index,
        logrank_p=logrank_p,
        metric_source=metric_source,
        dump_path=str(dump_path) if dump_path is not None else None,
        summary_path=str(summary_path) if summary_path is not None else None,
        log_path=str(log_path) if log_path is not None else None,
        notes=notes,
    )


def collect_results(results_root: Path) -> list[FoldMetrics]:
    fold_dirs = sorted(p for p in results_root.glob("k=*") if p.is_dir())
    metrics: list[FoldMetrics] = []
    for fold_dir in fold_dirs:
        name = fold_dir.name
        try:
            fold = int(name.split("=", 1)[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid fold directory name: {fold_dir}") from exc
        metrics.append(collect_fold_metrics(fold_dir, fold=fold))
    return metrics


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def summarize_results(
    metrics: list[FoldMetrics],
    *,
    expected_folds: int,
    min_mean_cindex: float,
    max_std: float,
    max_logrank_p: float,
    target_mean_cindex: float | None,
    target_mean_tolerance: float | None,
) -> dict[str, Any]:
    c_indexes = [m.c_index for m in metrics if m.c_index is not None]
    logrank_ps = [m.logrank_p for m in metrics if m.logrank_p is not None]
    best_fold = max((m for m in metrics if m.c_index is not None), key=lambda x: x.c_index, default=None)

    checks = {
        "fold_count_complete": len(metrics) >= int(expected_folds),
        "mean_c_index_ge_min": (_mean(c_indexes) is not None and _mean(c_indexes) >= float(min_mean_cindex)),
        "std_c_index_le_max": (_std(c_indexes) is not None and _std(c_indexes) <= float(max_std)),
        "all_logrank_p_lt_max": (bool(logrank_ps) and max(logrank_ps) < float(max_logrank_p)),
    }
    if target_mean_cindex is not None and target_mean_tolerance is not None:
        mean_cindex = _mean(c_indexes)
        checks["mean_c_index_close_to_target"] = (
            mean_cindex is not None and abs(mean_cindex - float(target_mean_cindex)) <= float(target_mean_tolerance)
        )

    passed_all = all(bool(v) for v in checks.values())
    status = "PASS" if passed_all else ("WARN" if c_indexes else "FAIL")

    return {
        "status": status,
        "expected_folds": int(expected_folds),
        "folds_found": len(metrics),
        "mean_c_index": _mean(c_indexes),
        "std_c_index": _std(c_indexes),
        "max_logrank_p": max(logrank_ps) if logrank_ps else None,
        "checks": checks,
        "best_fold_for_phase2": {
            "fold": best_fold.fold if best_fold is not None else None,
            "c_index": best_fold.c_index if best_fold is not None else None,
            "dump_path": best_fold.dump_path if best_fold is not None else None,
            "summary_path": best_fold.summary_path if best_fold is not None else None,
            "log_path": best_fold.log_path if best_fold is not None else None,
        },
        "folds": [
            {
                "fold": m.fold,
                "c_index": m.c_index,
                "logrank_p": m.logrank_p,
                "metric_source": m.metric_source,
                "dump_path": m.dump_path,
                "summary_path": m.summary_path,
                "log_path": m.log_path,
                "notes": m.notes,
            }
            for m in metrics
        ],
    }


def write_summary_csv(path: Path, metrics: list[FoldMetrics], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "c_index",
                "logrank_p",
                "metric_source",
                "dump_path",
                "summary_path",
                "log_path",
                "notes",
            ],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(
                {
                    "fold": m.fold,
                    "c_index": m.c_index,
                    "logrank_p": m.logrank_p,
                    "metric_source": m.metric_source,
                    "dump_path": m.dump_path,
                    "summary_path": m.summary_path,
                    "log_path": m.log_path,
                    "notes": " | ".join(m.notes),
                }
            )
        writer.writerow(
            {
                "fold": "aggregate",
                "c_index": summary.get("mean_c_index"),
                "logrank_p": summary.get("max_logrank_p"),
                "metric_source": summary.get("status"),
                "dump_path": "",
                "summary_path": "",
                "log_path": "",
                "notes": json.dumps(summary.get("checks", {}), ensure_ascii=False, sort_keys=True),
            }
        )
