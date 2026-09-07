from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .build_index import IndexRow
from .wsi_input import load_wsi_input, peek_wsi_num_tiles
from r2wsp.rna.gene_sets import GeneSetCollection, load_gene_sets_csv
from r2wsp.rna.tokenizer import OmicsSpec, build_omics_spec, tokenize_omics


@dataclass(frozen=True)
class TCGASample:
    case_id: str
    cohort: str
    wsi_feature_source: str
    wsi_feature_path: Path | tuple[Path, ...]
    wsi_feature_kind: str
    tile_tokens: torch.Tensor
    tile_xy: torch.Tensor
    tile_attn_mask: torch.Tensor
    rna_vec: torch.Tensor

    @property
    def token_source(self) -> str:
        return self.wsi_feature_source

    @property
    def token_path(self) -> Path:
        return self.wsi_feature_path


def _stable_split(case_id: str, *, seed: int, val_frac: float, test_frac: float) -> str:
    h = hash((str(case_id), int(seed))) & 0xFFFFFFFF
    r = (h % 1_000_000) / 1_000_000.0
    if r < float(test_frac):
        return "test"
    if r < float(test_frac) + float(val_frac):
        return "val"
    return "train"


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


def _strip_gene_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _allocate_case_shared_budgets(tile_counts: list[int], total_budget: int) -> list[int]:
    if not tile_counts:
        return []
    counts = [max(0, int(x)) for x in tile_counts]
    positive_indices = [i for i, c in enumerate(counts) if c > 0]
    if not positive_indices or int(total_budget) <= 0:
        return [0 for _ in counts]
    total_budget = int(total_budget)
    total_tiles = sum(counts)
    if total_tiles <= total_budget:
        return counts

    budgets = [0 for _ in counts]
    if total_budget >= len(positive_indices):
        for idx in positive_indices:
            budgets[idx] = 1
        remaining = total_budget - len(positive_indices)
    else:
        ranked = sorted(positive_indices, key=lambda i: counts[i], reverse=True)
        for idx in ranked[:total_budget]:
            budgets[idx] = 1
        return budgets

    residual_counts = [max(0, counts[i] - budgets[i]) for i in range(len(counts))]
    residual_total = sum(residual_counts)
    if remaining <= 0 or residual_total <= 0:
        return budgets

    fractional_parts: list[tuple[float, int]] = []
    assigned = 0
    for idx in positive_indices:
        raw_share = remaining * (float(residual_counts[idx]) / float(residual_total))
        extra = int(np.floor(raw_share))
        budgets[idx] += extra
        assigned += extra
        fractional_parts.append((raw_share - extra, idx))

    leftover = remaining - assigned
    for _, idx in sorted(fractional_parts, key=lambda x: (-x[0], -counts[x[1]]))[:leftover]:
        budgets[idx] += 1
    return [min(b, c) for b, c in zip(budgets, counts, strict=True)]


def _subsample_slide_tensors(
    tile_tokens: torch.Tensor,
    tile_xy: torch.Tensor,
    tile_attn_mask: torch.Tensor,
    keep_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_tiles = max(0, int(keep_tiles))
    total_tiles = int(tile_tokens.shape[0])
    if keep_tiles <= 0:
        return tile_tokens[:0], tile_xy[:0], tile_attn_mask[:0]
    if keep_tiles >= total_tiles:
        return tile_tokens, tile_xy, tile_attn_mask
    if keep_tiles == 1:
        indices = torch.tensor([0], dtype=torch.long)
    else:
        indices = torch.linspace(0, total_tiles - 1, steps=keep_tiles, dtype=torch.float32).round().long()
    return (
        tile_tokens.index_select(0, indices),
        tile_xy.index_select(0, indices),
        tile_attn_mask.index_select(0, indices),
    )


class AntiFeatureBank:
    """Loads precomputed anti-alignment features from
    ``{anti_feature_dir}/z_wsi_{cohort}.npy``, ``z_rna_{cohort}.npy`` and
    ``sample_ids_{cohort}.txt`` and returns them per-case.  Cases not present
    in the bank return zero vectors of ``default_dim`` so the rest of the
    pipeline always has a well-shaped tensor to ingest.
    """

    def __init__(
        self,
        anti_feature_dir: str | Path,
        cohorts: list[str] | str,
        *,
        default_dim: int = 128,
    ) -> None:
        self.dir = Path(anti_feature_dir)
        self.default_dim = int(default_dim)
        cohort_list = [str(c) for c in ([cohorts] if isinstance(cohorts, str) else cohorts)]
        self._wsi: dict[str, np.ndarray] = {}
        self._rna: dict[str, np.ndarray] = {}
        for cohort in cohort_list:
            z_wsi_p = self.dir / f"z_wsi_{cohort}.npy"
            z_rna_p = self.dir / f"z_rna_{cohort}.npy"
            ids_p = self.dir / f"sample_ids_{cohort}.txt"
            if not (z_wsi_p.is_file() and z_rna_p.is_file() and ids_p.is_file()):
                continue
            ids = ids_p.read_text().splitlines()
            zw = np.load(z_wsi_p)
            zr = np.load(z_rna_p)
            if zw.shape[0] != len(ids) or zr.shape[0] != len(ids):
                raise ValueError(f"anti-feature bank size mismatch for cohort={cohort}")
            if zw.ndim != 2 or zr.ndim != 2:
                raise ValueError(f"anti-feature bank expects 2D arrays for cohort={cohort}")
            self.default_dim = zw.shape[1]
            for k, arr in zip(ids, zw, strict=True):
                self._wsi[str(k)] = np.asarray(arr, dtype=np.float32)
            for k, arr in zip(ids, zr, strict=True):
                self._rna[str(k)] = np.asarray(arr, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self.default_dim

    def has_case(self, case_id: str) -> bool:
        return str(case_id) in self._wsi and str(case_id) in self._rna

    def get_wsi(self, case_id: str) -> torch.Tensor:
        arr = self._wsi.get(str(case_id))
        if arr is None:
            arr = np.zeros(self.default_dim, dtype=np.float32)
        return torch.from_numpy(arr.copy())

    def get_rna(self, case_id: str) -> torch.Tensor:
        arr = self._rna.get(str(case_id))
        if arr is None:
            arr = np.zeros(self.default_dim, dtype=np.float32)
        return torch.from_numpy(arr.copy())


class TCGAMultimodalDataset(Dataset):
    def __init__(
        self,
        rows: list[IndexRow],
        *,
        split: str | None = None,
        seed: int = 0,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        token_source: str | None = None,
        wsi_feature_source: str | None = None,
        max_tiles: int | None = None,
        tile_sampling: str = "prefix",
        tile_sampling_seed: int = 0,
        rna_mode: str = "vec",
        rna_gene_sets_csv: str | Path | None = None,
        rna_gene_id_to_symbol: dict[str, str] | None = None,
        use_multi_slide: bool = False,
        multi_slide_tile_budget_mode: str = "per_slide",
        use_anti_features: bool = False,
        anti_feature_dir: str | Path | None = None,
        anti_feature_cohorts: list[str] | str | None = None,
    ):
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError("split must be train|val|test|None")
        self.max_tiles = None if max_tiles is None else int(max_tiles)
        self.tile_sampling = str(tile_sampling)
        if self.tile_sampling not in {"prefix", "random"}:
            raise ValueError("tile_sampling must be prefix|random")
        self.tile_sampling_seed = int(tile_sampling_seed)
        self.rna_mode = str(rna_mode)
        self.use_multi_slide = bool(use_multi_slide)
        self.multi_slide_tile_budget_mode = str(multi_slide_tile_budget_mode)
        self.use_anti_features = bool(use_anti_features)
        if self.use_anti_features:
            if anti_feature_dir is None:
                raise ValueError("anti_feature_dir is required when use_anti_features=True")
            if anti_feature_cohorts is None:
                anti_feature_cohorts = sorted({str(r.cohort) for r in rows if r.cohort is not None})
            self._anti_bank: AntiFeatureBank | None = AntiFeatureBank(
                anti_feature_dir, anti_feature_cohorts,
            )
            self.anti_dim = self._anti_bank.dim
        else:
            self._anti_bank = None
            self.anti_dim = 0
        if self.rna_mode not in {"vec", "omics"}:
            raise ValueError("rna_mode must be vec|omics")
        if self.multi_slide_tile_budget_mode not in {"per_slide", "case_shared"}:
            raise ValueError("multi_slide_tile_budget_mode must be per_slide|case_shared")
        self._gene_sets: GeneSetCollection | None = None
        self._omics_spec: OmicsSpec | None = None
        self._rna_gene_id_to_symbol = rna_gene_id_to_symbol
        if self.rna_mode == "omics":
            if rna_gene_sets_csv is None:
                raise ValueError("rna_gene_sets_csv is required when rna_mode=omics")
            self._gene_sets = load_gene_sets_csv(rna_gene_sets_csv)

        source_filter = wsi_feature_source if wsi_feature_source is not None else token_source
        filtered: list[IndexRow] = []
        for r in rows:
            if r.case_id is None or r.cohort is None:
                continue
            if r.rna_path is None or r.clinical_path is None:
                continue
            if r.wsi_feature_path is None or not Path(r.wsi_feature_path).exists():
                continue
            if source_filter is not None and r.wsi_feature_source != source_filter:
                continue
            if split is not None:
                if _stable_split(r.case_id, seed=seed, val_frac=val_frac, test_frac=test_frac) != split:
                    continue
            filtered.append(r)

        if self.use_multi_slide:
            case_rows: dict[str, list[IndexRow]] = {}
            for r in filtered:
                case_rows.setdefault(str(r.case_id), []).append(r)
            self.rows: list[IndexRow | list[IndexRow]] = [case_rows[k] for k in sorted(case_rows.keys())]
        else:
            self.rows = filtered
        self.seed = int(seed)
        self._tpm_cache: dict[str, _CohortTPMTsv] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def get_omics_spec(self) -> OmicsSpec:
        if self.rna_mode != "omics" or self._gene_sets is None:
            raise ValueError("omics spec is only available when rna_mode=omics")
        if self._omics_spec is None:
            if not self.rows:
                raise ValueError("empty dataset")
            first_row = self.rows[0][0] if self.use_multi_slide else self.rows[0]
            self._ensure_omics_spec(Path(first_row.rna_path))
        if self._omics_spec is None:
            raise RuntimeError("failed to build omics spec")
        return self._omics_spec

    def _load_rna_vec(self, *, case_id: str, rna_path: Path) -> torch.Tensor:
        key = str(rna_path)
        if key not in self._tpm_cache:
            if rna_path.suffix.lower() != ".tsv":
                raise ValueError("only cohort-level TPM .tsv is supported in this skeleton")
            self._tpm_cache[key] = _CohortTPMTsv(rna_path)
        return self._tpm_cache[key].get_case_vec(case_id)

    def _ensure_omics_spec(self, rna_path: Path) -> None:
        if self._gene_sets is None:
            raise ValueError("gene sets not configured")
        key = str(rna_path)
        if key not in self._tpm_cache:
            self._tpm_cache[key] = _CohortTPMTsv(rna_path)
        gene_names = [
            self._rna_gene_id_to_symbol.get(_strip_gene_version(g), _strip_gene_version(g))
            if self._rna_gene_id_to_symbol is not None
            else str(g)
            for g in self._tpm_cache[key].genes
        ]
        spec = build_omics_spec(gene_names, self._gene_sets)
        if self._omics_spec is None:
            self._omics_spec = spec
            return
        if self._omics_spec.pathway_names != spec.pathway_names or self._omics_spec.omic_sizes != spec.omic_sizes:
            raise ValueError(f"inconsistent omics spec across cohorts: {rna_path}")

    def __getitem__(self, idx: int) -> dict[str, object]:
        if self.use_multi_slide:
            row_group = self.rows[idx]
            if not isinstance(row_group, list) or not row_group:
                raise ValueError("multi-slide dataset expects a non-empty list of rows per case")
            first = row_group[0]
            case_id = str(first.case_id)
            cohort = str(first.cohort)
            rna_path = Path(first.rna_path)
            slide_token_list: list[torch.Tensor] = []
            slide_xy_list: list[torch.Tensor] = []
            slide_mask_list: list[torch.Tensor] = []
            slide_ids_list: list[torch.Tensor] = []
            slide_paths: list[str] = []
            slide_rows = sorted(row_group, key=lambda x: (str(x.slide_id), str(x.wsi_feature_path)))
            if self.multi_slide_tile_budget_mode == "case_shared" and self.max_tiles is not None:
                tile_counts = [
                    peek_wsi_num_tiles(
                        Path(r.wsi_feature_path),
                        kind=r.wsi_feature_kind,
                        source_name=r.wsi_feature_source,
                    )
                    for r in slide_rows
                ]
                budgets = _allocate_case_shared_budgets(tile_counts, int(self.max_tiles))
            else:
                budgets = [self.max_tiles for _ in slide_rows]

            for slide_idx, (r, budget) in enumerate(zip(slide_rows, budgets, strict=True)):
                wsi_feature_path = Path(r.wsi_feature_path)
                tt, xy, am = load_wsi_input(
                    wsi_feature_path,
                    kind=r.wsi_feature_kind,
                    source_name=r.wsi_feature_source,
                    max_tiles=None if budget is None else int(budget),
                    tile_sampling=self.tile_sampling,
                    tile_sampling_seed=self.tile_sampling_seed,
                )
                n = int(tt.shape[0])
                if n <= 0:
                    continue
                slide_token_list.append(tt)
                slide_xy_list.append(xy)
                slide_mask_list.append(am)
                slide_ids_list.append(torch.full((n,), fill_value=int(slide_idx), dtype=torch.long))
                slide_paths.append(str(wsi_feature_path))
            if not slide_token_list:
                raise ValueError(f"case_shared budget removed all tiles for case_id={case_id}")
            tile_tokens = torch.cat(slide_token_list, dim=0)
            tile_xy = torch.cat(slide_xy_list, dim=0)
            tile_attn_mask = torch.cat(slide_mask_list, dim=0)
            slide_ids = torch.cat(slide_ids_list, dim=0)
            wsi_feature_source = str(first.wsi_feature_source)
            wsi_feature_kind = str(first.wsi_feature_kind)
            wsi_feature_path_out: str | list[str] = slide_paths
        else:
            r = self.rows[idx]
            if isinstance(r, list):
                raise ValueError("single-slide dataset expects IndexRow entries")
            case_id = str(r.case_id)
            cohort = str(r.cohort)
            wsi_feature_path = Path(r.wsi_feature_path)
            tile_tokens, tile_xy, tile_attn_mask = load_wsi_input(
                wsi_feature_path,
                kind=r.wsi_feature_kind,
                source_name=r.wsi_feature_source,
                max_tiles=self.max_tiles,
                tile_sampling=self.tile_sampling,
                tile_sampling_seed=self.tile_sampling_seed,
            )
            assert tile_tokens.shape[0] > 0, (
                f"no WSI tiles for case_id={case_id} wsi_feature_path={wsi_feature_path}"
            )
            slide_ids = torch.zeros((int(tile_tokens.shape[0]),), dtype=torch.long)
            rna_path = Path(r.rna_path)
            wsi_feature_source = str(r.wsi_feature_source)
            wsi_feature_kind = str(r.wsi_feature_kind)
            wsi_feature_path_out = str(wsi_feature_path)

        out: dict[str, object] = {
            "case_id": case_id,
            "cohort": cohort,
            "token_source": wsi_feature_source,
            "token_path": wsi_feature_path_out,
            "wsi_feature_source": wsi_feature_source,
            "wsi_feature_path": wsi_feature_path_out,
            "wsi_feature_kind": wsi_feature_kind,
            "tile_tokens": tile_tokens,
            "tile_xy": tile_xy,
            "tile_attn_mask": tile_attn_mask,
            "slide_ids": slide_ids,
        }
        if self.use_anti_features and self._anti_bank is not None:
            out["wsi_anti_vec"] = self._anti_bank.get_wsi(case_id)
            out["rna_anti_vec"] = self._anti_bank.get_rna(case_id)
        if self.rna_mode == "vec":
            out["rna_vec"] = self._load_rna_vec(case_id=case_id, rna_path=rna_path)
            return out
        self._ensure_omics_spec(rna_path)
        if self._omics_spec is None:
            raise RuntimeError("omics spec not initialized")
        rna_vec = self._load_rna_vec(case_id=case_id, rna_path=rna_path)
        out["rna_omics"] = tokenize_omics(rna_vec, self._omics_spec)
        return out
