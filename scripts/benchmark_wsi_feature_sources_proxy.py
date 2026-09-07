from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.build_index import IndexRow
from r2wsp.data.wsi_input import load_wsi_input


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_official_split(split_dir: Path, target_col: str) -> tuple[dict[str, dict[str, float]], set[str], set[str]]:
    censorship_col = target_col.split("_")[0] + "_censorship"
    train_rows = read_split_csv(split_dir / "train.csv")
    test_rows = read_split_csv(split_dir / "test.csv")
    case_table: dict[str, dict[str, float]] = {}
    for row in train_rows + test_rows:
        case_id = str(row["case_id"])
        case_table[case_id] = {
            "event_time": float(row[target_col]),
            "censorship": float(row[censorship_col]),
        }
    return case_table, {str(r["case_id"]) for r in train_rows}, {str(r["case_id"]) for r in test_rows}


def _random_split_train_val_case_ids(train_case_ids: list[str], seed: int, val_frac: float) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(train_case_ids), dtype=object)
    rng.shuffle(ids)
    val_size = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(str(x) for x in ids[:val_size].tolist())
    train_ids = set(str(x) for x in ids[val_size:].tolist())
    return train_ids, val_ids


def _build_survival_strata(
    train_case_ids: list[str],
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int,
) -> dict[str, list[str]]:
    df = pd.DataFrame(
        {
            "case_id": [str(c) for c in train_case_ids],
            "event_time": [float(case_table[str(c)]["event_time"]) for c in train_case_ids],
            "censorship": [float(case_table[str(c)]["censorship"]) for c in train_case_ids],
        }
    )
    uncensored = df.loc[df["censorship"] < 0.5, "event_time"].to_numpy(dtype=np.float64)
    if uncensored.size >= 2 and n_time_bins > 1:
        quantiles = np.quantile(uncensored, np.linspace(0.0, 1.0, int(n_time_bins) + 1))
        quantiles[0] = min(quantiles[0], float(df["event_time"].min())) - 1e-6
        quantiles[-1] = max(quantiles[-1], float(df["event_time"].max())) + 1e-6
        quantiles = np.unique(quantiles)
        if quantiles.size >= 3:
            df["time_bin"] = pd.cut(
                df["event_time"],
                bins=quantiles,
                labels=False,
                include_lowest=True,
            ).fillna(0).astype(int)
        else:
            df["time_bin"] = 0
    else:
        df["time_bin"] = 0
    df["stratum"] = df["censorship"].astype(int).astype(str) + "_" + df["time_bin"].astype(int).astype(str)
    strata: dict[str, list[str]] = {}
    for stratum, sub_df in df.groupby("stratum", sort=True):
        strata[str(stratum)] = [str(x) for x in sub_df["case_id"].tolist()]
    return strata


def split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int = 4,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    strata = _build_survival_strata(train_case_ids, case_table, n_time_bins=n_time_bins)
    val_ids: list[str] = []
    train_ids: list[str] = []
    for stratum in sorted(strata.keys()):
        ids = np.asarray(sorted(strata[stratum]), dtype=object)
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.extend(str(x) for x in ids.tolist())
            continue
        val_size = int(round(len(ids) * float(val_frac)))
        val_size = max(1, min(len(ids) - 1, val_size))
        val_ids.extend(str(x) for x in ids[:val_size].tolist())
        train_ids.extend(str(x) for x in ids[val_size:].tolist())
    if not val_ids or not train_ids:
        return _random_split_train_val_case_ids(train_case_ids, seed, val_frac)
    return set(train_ids), set(val_ids)


def neg_partial_log_likelihood(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(times, descending=True)
    risk = risk[order].reshape(-1)
    events = events[order].reshape(-1)
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    diff = risk - log_cumsum
    denom = events.sum().clamp_min(1.0)
    return -(diff * events).sum() / denom


def concordance_index(risk: np.ndarray, times: np.ndarray, censorships: np.ndarray) -> float:
    events = 1.0 - censorships.astype(np.float32)
    concordant = 0.0
    comparable = 0.0
    n = len(risk)
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and events[i] > 0:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] > 0:
                comparable += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


def source_rows_by_case(rows: list[IndexRow], source_name: str) -> dict[str, list[IndexRow]]:
    groups: dict[str, list[IndexRow]] = defaultdict(list)
    for row in rows:
        if row.case_id is None:
            continue
        if str(row.wsi_feature_source) != str(source_name):
            continue
        groups[str(row.case_id)].append(row)
    return groups


def pool_case_feature(
    row_group: list[IndexRow],
    *,
    max_tiles: int,
    tile_pool: str,
    case_pool: str,
) -> np.ndarray:
    slide_vecs: list[np.ndarray] = []
    for row in sorted(row_group, key=lambda x: (str(x.slide_id), str(x.wsi_feature_path))):
        tile_tokens, _, _ = load_wsi_input(
            Path(row.wsi_feature_path),
            kind=str(row.wsi_feature_kind),
            source_name=str(row.wsi_feature_source),
            max_tiles=int(max_tiles) if int(max_tiles) > 0 else None,
        )
        feat = tile_tokens.cpu().numpy().astype(np.float32, copy=False)
        if tile_pool == "mean":
            slide_vec = feat.mean(axis=0)
        elif tile_pool == "max":
            slide_vec = feat.max(axis=0)
        else:
            raise ValueError(f"unsupported tile_pool: {tile_pool}")
        slide_vecs.append(np.asarray(slide_vec, dtype=np.float32))
    if not slide_vecs:
        raise ValueError("empty row_group")
    slide_arr = np.stack(slide_vecs, axis=0)
    if case_pool == "mean":
        return slide_arr.mean(axis=0).astype(np.float32)
    if case_pool == "max":
        return slide_arr.max(axis=0).astype(np.float32)
    raise ValueError(f"unsupported case_pool: {case_pool}")


def build_case_matrix(
    rows: list[IndexRow],
    *,
    source_name: str,
    case_table: dict[str, dict[str, float]],
    max_tiles: int,
    tile_pool: str,
    case_pool: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    groups = source_rows_by_case(rows, source_name)
    case_ids = sorted([case_id for case_id in groups.keys() if case_id in case_table])
    if not case_ids:
        raise ValueError(f"no usable cases for source: {source_name}")
    features: list[np.ndarray] = []
    times: list[float] = []
    censorships: list[float] = []
    manifests: list[dict[str, object]] = []
    for case_id in case_ids:
        group = groups[case_id]
        vec = pool_case_feature(group, max_tiles=max_tiles, tile_pool=tile_pool, case_pool=case_pool)
        features.append(vec)
        times.append(float(case_table[case_id]["event_time"]))
        censorships.append(float(case_table[case_id]["censorship"]))
        manifests.append(
            {
                "case_id": case_id,
                "n_slides": len(group),
                "slide_ids": ";".join(str(r.slide_id) for r in sorted(group, key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))),
            }
        )
    return (
        np.stack(features, axis=0).astype(np.float32),
        np.asarray(times, dtype=np.float32),
        np.asarray(censorships, dtype=np.float32),
        case_ids,
        manifests,
    )


class LinearCoxHead(torch.nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.dropout = torch.nn.Dropout(float(dropout))
        self.linear = torch.nn.Linear(int(in_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x)).reshape(-1)


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_proxy_once(
    x: np.ndarray,
    times: np.ndarray,
    censorships: np.ndarray,
    case_ids: list[str],
    *,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    seed: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    device: torch.device,
) -> dict[str, object]:
    train_idx = np.asarray([i for i, case_id in enumerate(case_ids) if case_id in train_ids], dtype=np.int64)
    val_idx = np.asarray([i for i, case_id in enumerate(case_ids) if case_id in val_ids], dtype=np.int64)
    test_idx = np.asarray([i for i, case_id in enumerate(case_ids) if case_id in test_ids], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train/val/test split produced an empty partition")

    mean, std = fit_standardizer(x[train_idx])
    x_std = apply_standardizer(x, mean, std)
    x_t = torch.from_numpy(x_std).to(device=device, dtype=torch.float32)
    times_t = torch.from_numpy(times).to(device=device, dtype=torch.float32)
    censorships_t = torch.from_numpy(censorships).to(device=device, dtype=torch.float32)
    events_t = 1.0 - censorships_t

    set_seed(seed)
    model = LinearCoxHead(in_dim=int(x.shape[1]), dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val = float("-inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        risk_train = model(x_t[train_idx])
        loss = neg_partial_log_likelihood(risk_train, times_t[train_idx], events_t[train_idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            risk_all = model(x_t).detach().cpu().numpy().astype(np.float32)
        train_c = concordance_index(risk_all[train_idx], times[train_idx], censorships[train_idx])
        val_c = concordance_index(risk_all[val_idx], times[val_idx], censorships[val_idx])
        test_c = concordance_index(risk_all[test_idx], times[test_idx], censorships[test_idx])
        history.append(
            {
                "epoch": epoch,
                "train_c_index": train_c,
                "val_c_index": val_c,
                "test_c_index": test_c,
                "train_loss": float(loss.detach().cpu().item()),
            }
        )
        if np.isfinite(val_c) and float(val_c) > best_val:
            best_val = float(val_c)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        risk_all = model(x_t).detach().cpu().numpy().astype(np.float32)

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_c_index": concordance_index(risk_all[val_idx], times[val_idx], censorships[val_idx]),
        "best_test_c_index": concordance_index(risk_all[test_idx], times[test_idx], censorships[test_idx]),
        "best_train_c_index": concordance_index(risk_all[train_idx], times[train_idx], censorships[train_idx]),
        "train_case_count": int(len(train_idx)),
        "val_case_count": int(len(val_idx)),
        "test_case_count": int(len(test_idx)),
        "history": history,
    }


def summarize_runs(source_name: str, runs: list[dict[str, object]], *, n_cases: int, feat_dim: int) -> dict[str, object]:
    val_scores = [float(run["best_val_c_index"]) for run in runs]
    test_scores = [float(run["best_test_c_index"]) for run in runs]
    train_scores = [float(run["best_train_c_index"]) for run in runs]
    return {
        "source_name": source_name,
        "n_cases": int(n_cases),
        "feat_dim": int(feat_dim),
        "n_runs": int(len(runs)),
        "mean_best_val_c_index": float(np.mean(val_scores)),
        "std_best_val_c_index": float(np.std(val_scores)),
        "mean_best_test_c_index": float(np.mean(test_scores)),
        "std_best_test_c_index": float(np.std(test_scores)),
        "mean_best_train_c_index": float(np.mean(train_scores)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Independent proxy benchmark for WSI feature sources using a tiny pooled WSI-only Cox head."
    )
    p.add_argument("--project-root", default=str(_ROOT))
    p.add_argument("--data-root", default=None)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--target-col", default="dss_survival_days")
    p.add_argument("--sources", required=True, help="Comma separated WSI feature sources.")
    p.add_argument("--max-tiles", type=int, default=256)
    p.add_argument("--tile-pool", choices=["mean", "max"], default="mean")
    p.add_argument("--case-pool", choices=["mean", "max"], default="mean")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    paths = resolve_data_paths(project_root=project_root, data_root=args.data_root)
    paths.validate()
    inventory = scan_all_assets(paths)
    rows = build_index(inventory)
    case_table, train_case_ids, test_case_ids = load_official_split(split_dir, str(args.target_col))

    source_names = [x.strip() for x in str(args.sources).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    all_run_rows: list[dict[str, object]] = []
    source_summary_rows: list[dict[str, object]] = []
    source_meta: dict[str, object] = {}
    for source_name in source_names:
        x, times, censorships, case_ids, case_manifest = build_case_matrix(
            rows,
            source_name=source_name,
            case_table=case_table,
            max_tiles=int(args.max_tiles),
            tile_pool=str(args.tile_pool),
            case_pool=str(args.case_pool),
        )
        runs: list[dict[str, object]] = []
        for seed in seeds:
            train_ids, val_ids = split_train_val_case_ids(
                sorted(train_case_ids),
                int(seed),
                float(args.val_frac),
                case_table,
            )
            run = train_proxy_once(
                x,
                times,
                censorships,
                case_ids,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=set(test_case_ids),
                seed=int(seed),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                dropout=float(args.dropout),
                epochs=int(args.epochs),
                device=device,
            )
            run["source_name"] = source_name
            runs.append(run)
            all_run_rows.append(
                {
                    "source_name": source_name,
                    "seed": int(run["seed"]),
                    "best_epoch": int(run["best_epoch"]),
                    "best_train_c_index": float(run["best_train_c_index"]),
                    "best_val_c_index": float(run["best_val_c_index"]),
                    "best_test_c_index": float(run["best_test_c_index"]),
                    "train_case_count": int(run["train_case_count"]),
                    "val_case_count": int(run["val_case_count"]),
                    "test_case_count": int(run["test_case_count"]),
                }
            )

        summary = summarize_runs(source_name, runs, n_cases=len(case_ids), feat_dim=int(x.shape[1]))
        source_summary_rows.append(summary)
        source_histories = {
            f"seed_{int(run['seed'])}": run["history"] for run in runs
        }
        case_manifest_path = out_dir / f"{source_name}.case_manifest.csv"
        write_csv(case_manifest_path, case_manifest, list(case_manifest[0].keys()) if case_manifest else ["case_id"])
        history_path = out_dir / f"{source_name}.histories.json"
        save_json(history_path, source_histories)
        source_meta[source_name] = {
            "n_cases": len(case_ids),
            "feat_dim": int(x.shape[1]),
            "case_manifest_csv": str(case_manifest_path),
            "histories_json": str(history_path),
        }

    write_csv(
        out_dir / "proxy_run_results.csv",
        all_run_rows,
        list(all_run_rows[0].keys()) if all_run_rows else ["source_name"],
    )
    write_csv(
        out_dir / "proxy_source_summary.csv",
        source_summary_rows,
        list(source_summary_rows[0].keys()) if source_summary_rows else ["source_name"],
    )
    save_json(
        out_dir / "meta.json",
        {
            "project_root": str(project_root),
            "split_dir": str(split_dir),
            "target_col": str(args.target_col),
            "sources": source_names,
            "max_tiles": int(args.max_tiles),
            "tile_pool": str(args.tile_pool),
            "case_pool": str(args.case_pool),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "seeds": seeds,
            "device": str(device),
            "source_meta": source_meta,
        },
    )
    print(f"proxy_source_summary_csv={out_dir / 'proxy_source_summary.csv'}", flush=True)
    print(f"proxy_run_results_csv={out_dir / 'proxy_run_results.csv'}", flush=True)
    print(f"meta_json={out_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
