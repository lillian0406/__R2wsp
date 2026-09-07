from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import resolve_data_paths
from r2wsp.rna.gene_sets import load_gene_sets_csv
from r2wsp.rna.tokenizer import OmicsSpec, build_omics_spec, tokenize_omics


BRANCH_CHOICES = (
    "dense_vec",
    "omics",
    "omics_attn",
    "dense_omics",
    "dense_omics_gate",
    "dense_omics_residual",
    "dense_omics_cross_attn",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _survival_stratified_split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    n_time_bins: int,
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


def split_train_val_case_ids(
    train_case_ids: list[str],
    seed: int,
    val_frac: float,
    case_table: dict[str, dict[str, float]],
    *,
    mode: str,
    n_time_bins: int,
) -> tuple[set[str], set[str]]:
    if str(mode) == "survival_stratified":
        return _survival_stratified_split_train_val_case_ids(
            train_case_ids,
            seed,
            val_frac,
            case_table,
            n_time_bins=int(n_time_bins),
        )
    return _random_split_train_val_case_ids(train_case_ids, seed, val_frac)


def parse_csv_list(value: str | None, *, default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def infer_cohort_from_split_dir(split_dir: Path) -> str:
    m = re.search(r"TCGA_([A-Z0-9]+)_", str(split_dir).upper())
    if m is None:
        raise ValueError(f"cannot infer cohort from split_dir: {split_dir}")
    return str(m.group(1)).upper()


def resolve_default_rna_tsv(raw_rna_root: Path, cohort: str) -> Path:
    candidate = (raw_rna_root / "tpm_tsv" / f"{str(cohort).lower()}_tpm.tsv").resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"default RNA TSV not found for cohort={cohort}: {candidate}. "
        "Please pass --rna_tsv explicitly."
    )


def resolve_gene_sets_csv(raw_rna_root: Path, provided_path: str | None) -> Path:
    if provided_path is not None:
        path = Path(provided_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"rna gene sets csv not found: {path}")
        return path
    local_default = (raw_rna_root / "metadata" / "hallmarks_signatures.csv").resolve()
    if local_default.exists():
        return local_default
    raise FileNotFoundError(
        "No project-local hallmarks_signatures.csv found under data/raw_rna/metadata. "
        "Please pass --rna_gene_sets_csv explicitly."
    )


def strip_gene_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _parse_gtf_attributes(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(raw).strip().split(";"):
        item = chunk.strip()
        if not item:
            continue
        if " " not in item:
            continue
        key, value = item.split(" ", 1)
        out[str(key)] = str(value).strip().strip('"')
    return out


def load_gene_id_to_symbol_map(gtf_path: Path) -> dict[str, str]:
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    mapping: dict[str, str] = {}
    with opener(gtf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = _parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            gene_name = attrs.get("gene_name")
            if not gene_id or not gene_name:
                continue
            mapping[strip_gene_version(gene_id)] = str(gene_name)
    if not mapping:
        raise ValueError(f"no gene_id -> gene_name mapping found in GTF: {gtf_path}")
    return mapping


def resolve_gene_annotation_gtf(provided_path: str | None) -> Path | None:
    if provided_path is not None:
        path = Path(provided_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"gene_annotation_gtf not found: {path}")
        return path
    auto_candidates = [
        Path("/root/autodl-tmp/gencode.v22.annotation.gtf.gz"),
        Path("/root/autodl-tmp/gencode.v22.annotation.gtf"),
    ]
    for candidate in auto_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class _CohortTPMTsv:
    def __init__(self, path: Path):
        self.path = path
        df = pd.read_csv(self.path, sep="\t")
        if df.shape[1] < 3:
            raise ValueError("TPM TSV must have >= 3 columns (gene + >=2 samples)")
        self.gene_col = df.columns[0]
        df = df.set_index(self.gene_col)
        self.df = df
        self.genes = [str(g) for g in self.df.index.tolist()]
        self._case_to_column: dict[str, str] = {}

    @staticmethod
    def _sample_type_rank(column_name: str) -> tuple[int, str]:
        parts = str(column_name).split("-")
        if len(parts) >= 4:
            sample_code = parts[3][:2]
            if sample_code == "01":
                return (0, str(column_name))
            if sample_code.startswith("0"):
                return (1, str(column_name))
            if sample_code.startswith("1"):
                return (3, str(column_name))
        return (2, str(column_name))

    def _resolve_case_column(self, case_id: str) -> str:
        key = str(case_id)
        cached = self._case_to_column.get(key)
        if cached is not None:
            return cached
        if key in self.df.columns:
            self._case_to_column[key] = key
            return key

        candidates = [str(col) for col in self.df.columns if str(col).startswith(f"{key}-")]
        if not candidates:
            raise KeyError(f"case_id not found in TPM TSV: {key}")
        chosen = sorted(candidates, key=self._sample_type_rank)[0]
        self._case_to_column[key] = chosen
        return chosen

    def get_case_vec(self, case_id: str) -> torch.Tensor:
        col = self._resolve_case_column(case_id)
        arr = self.df[col].to_numpy(dtype=np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(arr).to(dtype=torch.float32)


@dataclass(frozen=True)
class RNABatch:
    case_id: list[str]
    rna_vec: torch.Tensor | None
    rna_omics: list[torch.Tensor] | None


class RNABaselineDataset(Dataset):
    def __init__(
        self,
        case_ids: list[str],
        *,
        rna_tsv_path: Path,
        branch: str,
        gene_sets_csv: Path | None = None,
        gene_id_to_symbol: dict[str, str] | None = None,
    ) -> None:
        self.case_ids = [str(x) for x in case_ids]
        self.branch = str(branch)
        if self.branch not in BRANCH_CHOICES:
            raise ValueError(f"unsupported branch: {branch}")
        self._tpm = _CohortTPMTsv(rna_tsv_path)
        self._omics_spec: OmicsSpec | None = None
        if self.branch in {"omics", "omics_attn", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"}:
            if gene_sets_csv is None:
                raise ValueError("gene_sets_csv is required for omics branches")
            gene_sets = load_gene_sets_csv(gene_sets_csv)
            gene_names = [
                gene_id_to_symbol.get(strip_gene_version(g), strip_gene_version(g)) if gene_id_to_symbol is not None else str(g)
                for g in self._tpm.genes
            ]
            self._omics_spec = build_omics_spec(gene_names, gene_sets)

    def __len__(self) -> int:
        return len(self.case_ids)

    @property
    def rna_dim(self) -> int:
        return int(len(self._tpm.genes))

    @property
    def omic_sizes(self) -> list[int] | None:
        if self._omics_spec is None:
            return None
        return self._omics_spec.omic_sizes

    @property
    def omics_spec(self) -> OmicsSpec | None:
        return self._omics_spec

    def __getitem__(self, idx: int) -> dict[str, object]:
        case_id = self.case_ids[idx]
        rna_vec = self._tpm.get_case_vec(case_id)
        out: dict[str, object] = {"case_id": case_id}
        if self.branch in {"dense_vec", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"}:
            out["rna_vec"] = rna_vec
        if self.branch in {"omics", "omics_attn", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"}:
            if self._omics_spec is None:
                raise RuntimeError("omics_spec not initialized")
            out["rna_omics"] = tokenize_omics(rna_vec, self._omics_spec)
        return out


def collate_rna_baseline(samples: list[dict[str, object]]) -> RNABatch:
    if not samples:
        raise ValueError("empty batch")
    case_ids = [str(s["case_id"]) for s in samples]
    rna_vec = None
    if "rna_vec" in samples[0]:
        rna_vec = torch.stack([s["rna_vec"] for s in samples], dim=0)
    rna_omics = None
    if "rna_omics" in samples[0]:
        omics0 = samples[0]["rna_omics"]
        if not isinstance(omics0, list):
            raise ValueError("rna_omics must be a list")
        rna_omics = []
        for i in range(len(omics0)):
            rna_omics.append(torch.stack([s["rna_omics"][i] for s in samples], dim=0))
    return RNABatch(case_id=case_ids, rna_vec=rna_vec, rna_omics=rna_omics)


def build_loader(
    case_ids: list[str],
    *,
    rna_tsv_path: Path,
    branch: str,
    gene_sets_csv: Path | None,
    gene_id_to_symbol: dict[str, str] | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[RNABaselineDataset, DataLoader]:
    ds = RNABaselineDataset(
        case_ids,
        rna_tsv_path=rna_tsv_path,
        branch=branch,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
        collate_fn=collate_rna_baseline,
    )
    return ds, loader


def _dense_block(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.LayerNorm(int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
    )


class OmicsMLPEncoder(nn.Module):
    def __init__(self, omic_sizes: list[int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.sig_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(s), int(hidden_dim)),
                    nn.ELU(),
                    nn.AlphaDropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.ELU(),
                    nn.AlphaDropout(float(dropout)),
                )
                for s in omic_sizes
            ]
        )
        self.attn_score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, rna_omics: list[torch.Tensor], pool_mode: str = "mean") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(rna_omics) != len(self.sig_networks):
            raise ValueError("rna_omics length mismatch")
        feats = [net(x.float()) for net, x in zip(self.sig_networks, rna_omics, strict=True)]
        tokens = torch.stack(feats, dim=1)
        if str(pool_mode) == "attn":
            attn_logits = self.attn_score(tokens).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            case = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        else:
            attn_weights = torch.full(
                (tokens.shape[0], tokens.shape[1]),
                1.0 / float(tokens.shape[1]),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            case = tokens.mean(dim=1)
        return case, tokens, attn_weights


class RNABranchSurvivalBaseline(nn.Module):
    def __init__(
        self,
        *,
        branch: str,
        rna_dim: int,
        hidden_dim: int,
        dropout: float,
        omic_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.branch = str(branch)
        if self.branch not in BRANCH_CHOICES:
            raise ValueError(f"unsupported branch: {branch}")

        self.dense_encoder = (
            _dense_block(rna_dim, hidden_dim, dropout)
            if self.branch in {"dense_vec", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"}
            else None
        )
        self.omics_encoder = (
            OmicsMLPEncoder(omic_sizes=omic_sizes or [], hidden_dim=hidden_dim, dropout=dropout)
            if self.branch in {"omics", "omics_attn", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"}
            else None
        )
        self.fusion_proj = (
            _dense_block(int(hidden_dim) * 2, hidden_dim, dropout)
            if self.branch in {"dense_omics", "dense_omics_gate", "dense_omics_cross_attn"}
            else None
        )
        self.residual_omics_proj = (
            nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
            )
            if self.branch == "dense_omics_residual"
            else None
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Sigmoid(),
            )
            if self.branch == "dense_omics_gate"
            else None
        )
        self.cross_attn = (
            nn.MultiheadAttention(int(hidden_dim), num_heads=4, dropout=float(dropout), batch_first=True)
            if self.branch == "dense_omics_cross_attn"
            else None
        )
        self.backbone = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        *,
        rna_vec: torch.Tensor | None = None,
        rna_omics: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        dense_case = None
        if self.dense_encoder is not None:
            if rna_vec is None:
                raise ValueError("rna_vec is required for dense branch")
            dense_case = self.dense_encoder(rna_vec.float())
            aux["rna_dense_case"] = dense_case

        omics_case = None
        if self.omics_encoder is not None:
            if rna_omics is None:
                raise ValueError("rna_omics is required for omics branch")
            pool_mode = "attn" if self.branch in {"omics_attn", "dense_omics_cross_attn"} else "mean"
            omics_case, omics_tokens, omics_attn_weights = self.omics_encoder(rna_omics, pool_mode=pool_mode)
            aux["rna_omics_case"] = omics_case
            aux["rna_omics_tokens"] = omics_tokens
            aux["rna_omics_attn_weights"] = omics_attn_weights

        if self.branch == "dense_vec":
            fused = dense_case
        elif self.branch in {"omics", "omics_attn"}:
            fused = omics_case
        elif self.branch == "dense_omics":
            if dense_case is None or omics_case is None or self.fusion_proj is None:
                raise RuntimeError("dense_omics requires both dense and omics features")
            fused = self.fusion_proj(torch.cat([dense_case, omics_case], dim=1))
        elif self.branch == "dense_omics_gate":
            if dense_case is None or omics_case is None or self.gate_mlp is None:
                raise RuntimeError("dense_omics_gate requires both dense and omics features")
            fusion_input = torch.cat([dense_case, omics_case], dim=1)
            gate = self.gate_mlp(fusion_input)
            mixed = gate * dense_case + (1.0 - gate) * omics_case
            aux["rna_fusion_gate"] = gate
            fused = self.fusion_proj(fusion_input) + mixed if self.fusion_proj is not None else mixed
        elif self.branch == "dense_omics_residual":
            if dense_case is None or omics_case is None or self.residual_omics_proj is None:
                raise RuntimeError("dense_omics_residual requires both dense and omics features")
            omics_res = self.residual_omics_proj(omics_case)
            fused = dense_case + omics_res
            aux["rna_omics_residual"] = omics_res
        elif self.branch == "dense_omics_cross_attn":
            if dense_case is None or omics_case is None or self.cross_attn is None or self.fusion_proj is None:
                raise RuntimeError("dense_omics_cross_attn requires both dense and omics features")
            query = dense_case.unsqueeze(1)
            attn_out, attn_weights = self.cross_attn(query=query, key=omics_tokens, value=omics_tokens)
            cross_case = attn_out.squeeze(1)
            aux["rna_cross_attn_case"] = cross_case
            aux["rna_cross_attn_weights"] = attn_weights
            fused = self.fusion_proj(torch.cat([dense_case, cross_case], dim=1))
        else:
            raise RuntimeError(f"unsupported forward branch: {self.branch}")

        if fused is None:
            raise RuntimeError("fused feature is None")
        hidden = self.backbone(fused)
        risk = self.head(hidden).squeeze(-1)
        aux["rna_fused"] = fused
        aux["rna_hidden"] = hidden
        return risk, aux


def gather_labels(case_ids: list[str], case_table: dict[str, dict[str, float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.tensor([case_table[c]["event_time"] for c in case_ids], dtype=torch.float32, device=device)
    censorships = torch.tensor([case_table[c]["censorship"] for c in case_ids], dtype=torch.float32, device=device)
    events = 1.0 - censorships
    return times, events


def evaluate(
    model: RNABranchSurvivalBaseline,
    loader: DataLoader,
    case_table: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    risks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    censorships: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_times_t, batch_events_t = gather_labels(batch.case_id, case_table, device)
            risk, _ = model(
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
            )
            batch_loss = neg_partial_log_likelihood(risk, batch_times_t, batch_events_t)
            batch_risk = risk.detach().cpu().numpy()
            batch_times = np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32)
            batch_censorships = np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32)
            losses.append(float(batch_loss.detach().cpu()))
            risks.append(batch_risk)
            times.append(batch_times)
            censorships.append(batch_censorships)
    risk_all = np.concatenate(risks, axis=0)
    time_all = np.concatenate(times, axis=0)
    censorship_all = np.concatenate(censorships, axis=0)
    return concordance_index(risk_all, time_all, censorship_all), float(np.mean(losses)), risk_all


def train_one_run(
    *,
    branch: str,
    seed: int,
    split_dir: Path,
    target_col: str,
    rna_tsv_path: Path,
    gene_sets_csv: Path | None,
    gene_id_to_symbol: dict[str, str] | None,
    out_dir: Path,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    val_split_mode: str,
    val_time_bins: int,
    selection_metric: str,
    val_ema_decay: float,
    selection_min_epochs: int,
    early_stop_patience: int,
    device_name: str,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, object]:
    set_seed(seed)
    case_table, official_train_case_ids, test_case_ids = load_official_split(split_dir, target_col)
    train_case_ids, val_case_ids = split_train_val_case_ids(
        sorted(official_train_case_ids),
        int(seed),
        float(val_frac),
        case_table,
        mode=str(val_split_mode),
        n_time_bins=int(val_time_bins),
    )

    train_ds, train_loader = build_loader(
        sorted(train_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch=branch,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_ds, val_loader = build_loader(
        sorted(val_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch=branch,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_ds, test_loader = build_loader(
        sorted(test_case_ids),
        rna_tsv_path=rna_tsv_path,
        branch=branch,
        gene_sets_csv=gene_sets_csv,
        gene_id_to_symbol=gene_id_to_symbol,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = RNABranchSurvivalBaseline(
        branch=branch,
        rna_dim=int(train_ds.rna_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        omic_sizes=train_ds.omic_sizes,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_val = float("-inf")
    best_test = float("nan")
    best_train = float("nan")
    best_val_loss = float("inf")
    best_select_score = float("-inf") if str(selection_metric) != "val_loss" else float("inf")
    val_c_index_ema: float | None = None
    stale_epochs = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)

    print(f"[run] branch={branch} seed={seed} out_dir={out_dir}")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_risks: list[np.ndarray] = []
        train_times_np: list[np.ndarray] = []
        train_cens_np: list[np.ndarray] = []
        for batch in train_loader:
            times, events = gather_labels(batch.case_id, case_table, device)
            risk, _ = model(
                rna_vec=batch.rna_vec.to(device) if batch.rna_vec is not None else None,
                rna_omics=[x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None,
            )
            loss = neg_partial_log_likelihood(risk, times, events)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_risks.append(risk.detach().cpu().numpy())
            train_times_np.append(np.asarray([case_table[c]["event_time"] for c in batch.case_id], dtype=np.float32))
            train_cens_np.append(np.asarray([case_table[c]["censorship"] for c in batch.case_id], dtype=np.float32))

        train_c_index = concordance_index(
            np.concatenate(train_risks, axis=0),
            np.concatenate(train_times_np, axis=0),
            np.concatenate(train_cens_np, axis=0),
        )
        val_c_index, val_loss, _ = evaluate(model, val_loader, case_table, device)
        test_c_index, _, _ = evaluate(model, test_loader, case_table, device)

        if val_c_index_ema is None:
            val_c_index_ema = float(val_c_index)
        else:
            decay = float(val_ema_decay)
            val_c_index_ema = decay * float(val_c_index_ema) + (1.0 - decay) * float(val_c_index)

        if str(selection_metric) == "val_loss":
            select_score = float(val_loss)
            improved = epoch >= int(selection_min_epochs) and (
                select_score < best_select_score if np.isfinite(best_select_score) else True
            )
        elif str(selection_metric) == "val_c_index_ema":
            select_score = float(val_c_index_ema)
            improved = epoch >= int(selection_min_epochs) and select_score >= float(best_select_score)
        else:
            select_score = float(val_c_index)
            improved = epoch >= int(selection_min_epochs) and select_score >= float(best_select_score)

        if improved:
            best_val = float(val_c_index)
            best_val_loss = float(val_loss)
            best_test = float(test_c_index)
            best_train = float(train_c_index)
            best_epoch = epoch
            best_select_score = float(select_score)
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "branch": branch,
                    "seed": int(seed),
                    "best_epoch": best_epoch,
                    "best_val_c_index": best_val,
                    "best_val_loss": best_val_loss,
                    "best_test_c_index": best_test,
                    "best_select_score": best_select_score,
                },
                out_dir / "best.pt",
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "loss": float(np.mean(train_losses)),
            "train_c_index": float(train_c_index),
            "val_c_index": float(val_c_index),
            "val_loss": float(val_loss),
            "val_c_index_ema": float(val_c_index_ema),
            "selection_metric": str(selection_metric),
            "selection_score": float(select_score),
            "test_c_index": float(test_c_index),
        }
        history_rows.append(row)
        print(
            f"epoch={epoch} branch={branch} seed={seed} loss={row['loss']:.4f} "
            f"train_c_index={row['train_c_index']:.4f} val_c_index={row['val_c_index']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_c_index_ema={row['val_c_index_ema']:.4f} "
            f"test_c_index={row['test_c_index']:.4f}"
        )
        if epoch >= int(selection_min_epochs) and stale_epochs >= int(early_stop_patience):
            print(
                f"early_stop branch={branch} seed={seed} epoch={epoch} "
                f"selection_metric={str(selection_metric)} best_epoch={best_epoch} stale_epochs={stale_epochs}"
            )
            break

    write_csv(
        out_dir / "histories" / f"seed{int(seed)}.csv",
        history_rows,
        ["epoch", "loss", "train_c_index", "val_c_index", "val_loss", "val_c_index_ema", "selection_metric", "selection_score", "test_c_index"],
    )

    summary = {
        "mode": "rna_survival_baseline",
        "branch": branch,
        "seed": int(seed),
        "split_dir": str(split_dir),
        "target_col": str(target_col),
        "rna_tsv_path": str(rna_tsv_path),
        "rna_gene_sets_csv": str(gene_sets_csv) if gene_sets_csv is not None else None,
        "gene_annotation_gtf_used": bool(gene_id_to_symbol is not None),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "n_train_cases": len(train_ds),
        "n_val_cases": len(val_ds),
        "n_test_cases": len(test_ds),
        "selection_metric": str(selection_metric),
        "best_epoch": int(best_epoch),
        "best_val_c_index": float(best_val),
        "best_val_loss": float(best_val_loss),
        "best_select_score": float(best_select_score),
        "best_test_c_index": float(best_test),
        "best_train_c_index": float(best_train),
        "final_test_c_index": history_rows[-1]["test_c_index"] if history_rows else float("nan"),
        "out_dir": str(out_dir),
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="RNA-only survival baseline sweep for dense, omics, and dense+omics branches.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--target_col", default="dss_survival_days")
    p.add_argument("--rna_tsv", default=None)
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    p.add_argument("--branches", default="dense_vec,omics,dense_omics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--val_split_mode", default="survival_stratified", choices=["random", "survival_stratified"])
    p.add_argument("--val_time_bins", type=int, default=4)
    p.add_argument("--selection_metric", default="val_c_index_ema", choices=["val_c_index", "val_c_index_ema", "val_loss"])
    p.add_argument("--val_ema_decay", type=float, default=0.6)
    p.add_argument("--selection_min_epochs", type=int, default=8)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--run_tag", default="baseline_sweep")
    p.add_argument("--out_dir", default="outputs/rna_branch_baseline")
    args = p.parse_args()

    split_dir = Path(args.split_dir).resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"split_dir not found: {split_dir}")
    branches = parse_csv_list(args.branches, default=list(BRANCH_CHOICES))
    invalid_branches = [x for x in branches if x not in BRANCH_CHOICES]
    if invalid_branches:
        raise ValueError(f"unsupported branches: {invalid_branches}")

    seed_values = parse_csv_list(args.seeds, default=[str(int(args.seed))])
    seeds = [int(x) for x in seed_values]

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()
    cohort = infer_cohort_from_split_dir(split_dir)
    rna_tsv_path = Path(args.rna_tsv).resolve() if args.rna_tsv is not None else resolve_default_rna_tsv(paths.raw_rna_root, cohort)
    if not rna_tsv_path.exists():
        raise FileNotFoundError(f"rna_tsv not found: {rna_tsv_path}")

    gene_sets_csv = None
    gene_id_to_symbol = None
    if any(branch in {"omics", "omics_attn", "dense_omics", "dense_omics_gate", "dense_omics_residual", "dense_omics_cross_attn"} for branch in branches):
        gene_sets_csv = resolve_gene_sets_csv(paths.raw_rna_root, args.rna_gene_sets_csv)
        gtf_path = resolve_gene_annotation_gtf(args.gene_annotation_gtf)
        if gtf_path is not None:
            gene_id_to_symbol = load_gene_id_to_symbol_map(gtf_path)
        else:
            print("[warn] no gene_annotation_gtf found, omics matching will use raw gene ids")

    run_root = (Path(args.out_dir).resolve() / split_dir.name / str(args.run_tag)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] split_dir={split_dir}")
    print(f"[sweep] rna_tsv_path={rna_tsv_path}")
    print(f"[sweep] gene_sets_csv={gene_sets_csv}")
    print(f"[sweep] gene_annotation_gtf={resolve_gene_annotation_gtf(args.gene_annotation_gtf)}")
    print(f"[sweep] run_root={run_root}")

    leaderboard_rows: list[dict[str, object]] = []
    for branch in branches:
        for seed in seeds:
            out_dir = run_root / f"branch={branch}" / f"seed={int(seed)}"
            summary = train_one_run(
                branch=branch,
                seed=int(seed),
                split_dir=split_dir,
                target_col=str(args.target_col),
                rna_tsv_path=rna_tsv_path,
                gene_sets_csv=gene_sets_csv,
                gene_id_to_symbol=gene_id_to_symbol,
                out_dir=out_dir,
                hidden_dim=int(args.hidden_dim),
                dropout=float(args.dropout),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                val_frac=float(args.val_frac),
                val_split_mode=str(args.val_split_mode),
                val_time_bins=int(args.val_time_bins),
                selection_metric=str(args.selection_metric),
                val_ema_decay=float(args.val_ema_decay),
                selection_min_epochs=int(args.selection_min_epochs),
                early_stop_patience=int(args.early_stop_patience),
                device_name=str(args.device),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
            )
            leaderboard_rows.append(summary)
            leaderboard_fieldnames = [
                "branch",
                "seed",
                "best_epoch",
                "best_val_c_index",
                "best_test_c_index",
                "best_train_c_index",
                "best_val_loss",
                "best_select_score",
                "n_train_cases",
                "n_val_cases",
                "n_test_cases",
                "out_dir",
            ]
            leaderboard_export = [
                {key: row.get(key) for key in leaderboard_fieldnames}
                for row in sorted(
                    leaderboard_rows,
                    key=lambda x: (float(x["best_val_c_index"]), float(x["best_test_c_index"])),
                    reverse=True,
                )
            ]
            write_csv(
                run_root / "leaderboard.csv",
                leaderboard_export,
                leaderboard_fieldnames,
            )

    save_json(
        run_root / "summary.json",
        {
            "mode": "rna_survival_baseline_sweep",
            "split_dir": str(split_dir),
            "run_root": str(run_root),
            "rna_tsv_path": str(rna_tsv_path),
            "gene_sets_csv": str(gene_sets_csv) if gene_sets_csv is not None else None,
            "gene_annotation_gtf": str(resolve_gene_annotation_gtf(args.gene_annotation_gtf)) if resolve_gene_annotation_gtf(args.gene_annotation_gtf) is not None else None,
            "branches": branches,
            "seeds": seeds,
            "runs": leaderboard_rows,
        },
    )
    print(f"[done] run_root={run_root}")


if __name__ == "__main__":
    main()
