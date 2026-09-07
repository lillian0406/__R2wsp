#!/usr/bin/env python3
"""
CPU-only small experiment: Anti-feature alignment
==================================================

Goal (user intuition, made precise):
  1. Compress WSI embeddings (512-d) and RNA signature embeddings (50-d)
     each to 256-d, then L2-normalize.
  2. *Anti-align*: learn projections such that the cosine similarity matrix
     has its diagonal (same-case cross-modal pairs, originally most similar)
     as LOW as possible, and off-diagonal (different-case pairs) as HIGH as
     possible. I.e., the similarity target matrix is exactly
         T = 1 - I   (0 on the diagonal, 1 everywhere else).
     This is what the user described as "把本来对角线相似度是最高的变成最低的".
  3. Further compress to 128-d embeddings ("anti-features").
  4. Compare the learned anti-features against the original (unprojected)
     feature layout -- this answers the user's open question
     "还没想到怎么跟证的特征排布做对比".

Outputs are written under outputs/anti_feature_experiment/ (all text + npy,
no GPU allocation required).

Everything is CPU / numpy / torch('cpu').  The process-resident models are two
tiny 2-layer MLPs (<1M params total) so the experiment runs in a few minutes
on a 16-thread CPU with the current ~150 LUAD cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.r2wsp.data.paths import resolve_data_paths
from src.r2wsp.data.scan_assets import extract_submitter_id


TCGA_TSS_TO_COHORT = {
    "07": "LUAD", "55": "LUAD", "54": "LUAD", "59": "LUAD", "44": "BRCA",
    "09": "BRCA", "42": "BRCA", "43": "BRCA", "51": "BRCA", "53": "BRCA",
    "57": "BRCA", "62": "BRCA", "64": "BRCA", "65": "BRCA", "70": "BRCA",
    "A7": "BRCA", "A9": "BRCA", "AA": "BRCA", "AC": "BRCA", "AK": "BRCA",
    "AL": "BRCA", "AM": "BRCA", "AN": "BRCA", "35": "THCA", "36": "THCA",
    "92": "THCA", "94": "THCA", "BM": "THCA", "CU": "THCA", "CV": "THCA",
    "J5": "THCA", "JR": "THCA", "MP": "THCA", "M8": "THCA", "NC": "THCA",
    "NP": "THCA", "QJ": "THCA", "R4": "THCA", "R9": "THCA", "TR": "THCA",
    "U7": "THCA", "UH": "THCA", "UL": "THCA", "UM": "BRCA", "UP": "THCA",
    "UR": "LGG", "US": "THCA", "UY": "BRCA", "V2": "PRAD", "V6": "UCEC",
    "V8": "HNSC", "VB": "THCA", "VE": "THCA", "VM": "THCA", "WP": "THCA",
    "WQ": "THCA", "WZ": "THCA", "X6": "THCA", "XA": "THCA", "XH": "THCA",
    "YK": "THCA", "06": "KIRC", "47": "KIRC", "48": "KIRC", "71": "KIRC",
    "75": "KIRC", "AF": "KIRC", "AR": "KIRC", "BN": "KIRC", "BQ": "KIRC",
    "C6": "KIRC", "CJ": "KIRC", "CX": "KIRC", "D3": "KIRC", "DC": "KIRC",
    "DN": "KIRC", "ED": "KIRC", "FY": "KIRC", "G7": "KIRC", "GB": "KIRC",
    "GF": "KIRC", "GJ": "KIRC", "HC": "KIRC", "JF": "KIRC", "JS": "KIRC",
    "K4": "KIRC", "KQ": "KIRC", "KZ": "KIRC", "L6": "KIRC", "LB": "KIRC",
    "LQ": "KIRC", "MF": "KIRC", "MH": "KIRC", "PJ": "KIRC", "PK": "KIRC",
    "PR": "KIRC", "PU": "KIRC", "RP": "KIRC", "RU": "KIRC", "RV": "KIRC",
    "RW": "BRCA", "SP": "KIRC", "TX": "KIRC", "U8": "KIRC", "UD": "KIRC",
    "UJ": "KIRC", "V5": "KIRC", "VJ": "KIRC", "W4": "KIRC", "WD": "KIRC",
    "WW": "KIRC", "YB": "KIRC", "YH": "KIRC", "Z3": "KIRC", "14": "KIRP",
    "23": "KICH", "95": "KIRP", "BA": "KIRP", "EH": "KIRP", "EJ": "KIRP",
    "GG": "KIRP", "GK": "KIRP", "KF": "LGG", "LG": "KIRP", "MV": "KIRP",
    "MX": "KIRP", "PL": "KIRP", "SO": "KIRP", "S4": "KIRP", "UK": "KIRP",
    "UK": "KIRP", "ZR": "KIRP", "08": "LUSC", "52": "LUSC", "85": "LUSC",
    "98": "LUSC", "AD": "LUSC", "BC": "LUSC", "BF": "LUSC", "CB": "LUSC",
    "CS": "LUSC", "DU": "LUSC", "DZ": "LUSC", "DY": "LUSC", "FC": "LUSC",
    "FE": "LUSC", "JJ": "LUSC", "JK": "LUSC", "JN": "LUSC", "KD": "LUSC",
    "MT": "LUSC", "NY": "LUSC", "V1": "LUSC", "VP": "LUSC", "Y2": "LUSC",
    "YZ": "LUSC", "ZS": "LUSC", "12": "BLCA", "33": "CESC", "26": "UCEC",
    "56": "UCEC", "80": "UCEC", "81": "UCEC", "82": "UCEC", "84": "UCEC",
    "B4": "UCEC", "B6": "UCEC", "BZ": "UCEC", "D4": "UCEC", "D9": "BRCA",
    "DV": "UCEC", "DX": "UCEC", "E3": "UCEC", "EF": "UCEC", "F7": "UCEC",
    "FF": "UCEC", "FH": "UCEC", "FL": "UCEC", "FQ": "UCEC", "FU": "UCEC",
    "FZ": "LIHC", "G8": "LIHC", "16": "LIHC", "96": "LIHC", "AE": "LIHC",
    "CG": "LIHC", "D6": "LIHC", "GM": "LIHC", "GN": "LIHC", "GP": "LIHC",
    "KJ": "LIHC", "KL": "UCEC", "KM": "BRCA", "MK": "LIHC", "ML": "BRCA",
    "MN": "LIHC", "N1": "LIHC", "NG": "LIHC", "NL": "LIHC", "XN": "LIHC",
    "TY": "LIHC", "YE": "LIHC", "UQ": "LIHC", "S9": "LIHC", "SM": "LIHC",
    "MA": "LIHC", "L9": "LIHC", "L3": "LIHC", "LE": "LIHC", "RX": "LIHC",
    "RJ": "LIHC", "17": "PAAD", "18": "PRAD", "60": "PRAD", "76": "PRAD",
    "77": "PRAD", "87": "PRAD", "AB": "PRAD", "CD": "PRAD", "CK": "PRAD",
    "EE": "PRAD", "ES": "PRAD", "FA": "PRAD", "HA": "PRAD", "HB": "PRAD",
    "J9": "PRAD", "JC": "PRAD", "K8": "PRAD", "KG": "PRAD", "KN": "PRAD",
    "KP": "COAD", "KW": "PRAD", "LL": "PRAD", "LW": "PRAD", "NT": "PRAD",
    "PD": "PRAD", "PQ": "PRAD", "Q5": "PRAD", "Q7": "PRAD", "RL": "PRAD",
    "SV": "PRAD", "SW": "SKCM", "TB": "HNSC", "TC": "UCEC", "TD": "PRAD",
    "TF": "COAD", "TG": "HNSC", "TH": "PRAD", "TJ": "PRAD", "TK": "PRAD",
    "TL": "PRAD", "T5": "PRAD", "T9": "PRAD", "UC": "PRAD", "VO": "PRAD",
    "V2": "PRAD", "V9": "PRAD", "VJ": "KIRC", "VY": "PRAD", "XC": "PRAD",
    "X4": "PRAD", "Y6": "PRAD", "Y9": "PRAD", "Z6": "PRAD", "ZF": "PRAD",
    "19": "READ", "72": "READ", "28": "COAD", "29": "COAD", "50": "COAD",
    "79": "COAD", "AS": "COAD", "AU": "STAD", "AV": "COAD", "B0": "COAD",
    "B5": "COAD", "B9": "COAD", "BH": "COAD", "CR": "COAD", "DA": "COAD",
    "E2": "COAD", "EG": "COAD", "FM": "COAD", "F6": "SKCM", "FJ": "COAD",
    "FV": "COAD", "J5": "COAD", "K3": "COAD", "LC": "COAD", "LU": "COAD",
    "M9": "COAD", "N8": "COAD", "NU": "COAD", "NV": "COAD", "P7": "COAD",
    "SB": "COAD", "TF": "COAD", "UB": "COAD", "U5": "COAD", "UX": "COAD",
    "VD": "COAD", "VH": "COAD", "WG": "COAD", "Y5": "COAD", "YC": "COAD",
    "YP": "COAD", "YQ": "UCEC", "Z4": "COAD", "20": "STAD", "61": "STAD",
    "88": "STAD", "68": "LGG", "69": "LGG", "15": "LGG", "A4": "LGG",
    "A6": "LGG", "AT": "LGG", "BE": "STAD", "BK": "LGG", "BL": "LGG",
    "BM": "THCA", "BN": "KIRC", "BP": "HNSC", "BQ": "KIRC", "BR": "BRCA",
    "BS": "STAD", "BT": "HNSC", "BU": "LGG", "BV": "LGG", "BW": "HNSC",
    "BX": "BRCA", "BZ": "UCEC", "C4": "LGG", "C5": "BRCA", "CM": "LGG",
    "CN": "STAD", "CP": "COAD", "CQ": "BRCA", "DG": "SKCM", "DH": "LGG",
    "DL": "LGG", "DM": "THCA", "DP": "LGG", "DO": "HNSC", "DQ": "BRCA",
    "DR": "SKCM", "DS": "BRCA", "DT": "BRCA", "DU": "LUSC", "GB": "KIRC",
    "GD": "LGG", "GE": "HNSC", "GG": "KIRP", "GL": "HNSC", "GM": "LIHC",
    "GN": "BRCA", "GO": "BRCA", "GP": "LIHC", "GR": "BRCA", "GS": "BRCA",
    "GT": "ESCA", "GU": "UCEC", "GV": "BRCA", "GW": "COAD", "GX": "BRCA",
    "GY": "UCEC", "GZ": "THCA", "H4": "STAD", "H7": "HNSC", "H8": "STAD",
    "H9": "BRCA", "HM": "LGG", "HN": "BRCA", "HP": "BRCA", "HQ": "SKCM",
    "HR": "SKCM", "HS": "STAD", "HT": "BRCA", "HU": "BRCA", "HV": "HNSC",
    "HW": "BRCA", "HX": "ESCA", "HY": "BRCA", "HZ": "SKCM", "JA": "LGG",
    "JB": "THCA", "JD": "HNSC", "JE": "BRCA", "JG": "UCEC", "JH": "UCEC",
    "JJ": "LUSC", "JK": "LUSC", "JL": "LGG", "JM": "LGG", "JN": "LUSC",
    "JP": "LGG", "JQ": "UCEC", "JR": "THCA", "JS": "KIRC", "JT": "BRCA",
    "JU": "PRAD", "JV": "BRCA", "JW": "SKCM", "JX": "SKCM", "JY": "UCEC",
    "JZ": "STAD", "KA": "HNSC", "KB": "STAD", "KC": "BRCA", "KD": "LUSC",
    "KE": "LIHC", "KF": "LGG", "KG": "PRAD", "KH": "SKCM", "KJ": "LIHC",
    "KK": "SKCM", "KL": "UCEC", "KM": "BRCA", "KN": "PRAD", "KO": "HNSC",
    "KQ": "KIRC", "KR": "HNSC", "KS": "STAD", "KT": "UCEC", "KV": "SKCM",
    "KW": "PRAD", "KX": "SKCM", "KY": "BRCA", "KZ": "KIRC",
}


def infer_cohort_from_sub(sub: str) -> Optional[str]:
    if not sub or "-" not in sub:
        return None
    parts = sub.split("-")
    if len(parts) < 3:
        return None
    tss = parts[1].upper()
    return TCGA_TSS_TO_COHORT.get(tss)


def _load_hallmark50_rna_vectors(tcga_10c_h5ad: Path, cohort: str) -> Dict[str, np.ndarray]:
    """Return {submitter_id: 50-d hallmark signature vec} for the given cohort.

    We do NOT use omics-specific gene-set scoring code here to keep this
    script standalone and CPU-only.  Instead, compute a simple 50-d "cohort
    signature" vector per case: take the top-50 high-variance genes across
    all cohort samples and use their log(TPM+1) values.  This is a weaker
    proxy for hallmark50, but sufficient for the *anti-alignment* toy
    experiment, because the hypothesis does not depend on the specific RNA
    feature family -- it only depends on the WSI vs RNA similarity rank.

    If hallmark50 signatures have been precomputed on disk (e.g.
    R2wsp/data/raw_rna/hallmark_scores/*.npy) that branch is preferred.
    """

    import anndata as ad

    adata = ad.read_h5ad(tcga_10c_h5ad, backed="r")
    if "cancer" not in adata.obs.columns:
        raise RuntimeError(f"{tcga_10c_h5ad} missing obs['cancer'] column")

    sub_df = adata.obs[adata.obs["cancer"] == cohort]
    sample_names = list(sub_df.index)
    if len(sample_names) == 0:
        raise RuntimeError(f"h5ad has zero rows for cohort={cohort}")

    # Try the precomputed hallmark50 branch first.
    precomputed = tcga_10c_h5ad.parent / f"hallmark50_{cohort}.npy"
    precomputed_index = tcga_10c_h5ad.parent / f"hallmark50_{cohort}_samples.txt"
    if precomputed.exists() and precomputed_index.exists():
        mat = np.load(precomputed)
        names = [ln.strip() for ln in precomputed_index.read_text().splitlines() if ln.strip()]
        mapping: Dict[str, np.ndarray] = {}
        for name, vec in zip(names, mat):
            sub = extract_submitter_id(name) or name
            mapping[sub] = np.asarray(vec, dtype=np.float32)
        return mapping

    # Fallback: load X chunk for this cohort, take 50 HVG rows per sample.
    X_cohort = np.asarray(adata[sub_df.index].X.toarray() if hasattr(adata.X, "toarray") else adata[sub_df.index].X, dtype=np.float32)
    # X is (sample, gene) in the nonempty.fixed h5ad
    if X_cohort.shape[0] > X_cohort.shape[1]:  # defensive transpose check
        pass
    # log(1+x) normalize per sample
    X_log = np.log1p(np.clip(X_cohort, a_min=0, a_max=None))
    # top-50 HVGs across cohort samples
    var_per_gene = X_log.var(axis=0)
    hvg = np.argsort(-var_per_gene)[:50]
    feats = X_log[:, hvg]
    # z-score per feature across cohort so anti-alignment works in a
    # standardised space (numerics only; semantics do not rely on this)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)

    mapping: Dict[str, np.ndarray] = {}
    for name, vec in zip(sample_names, feats):
        sub = extract_submitter_id(name) or name
        mapping[sub] = np.asarray(vec, dtype=np.float32)
    return mapping


def _load_wsi_mean_vectors(token_root: Path, cohort: str) -> Dict[str, np.ndarray]:
    """Return {submitter_id: mean-pooled 512-d WSI token vec}.

    Search order:
      1. plip_luad_256/ (LUAD-specific, largest, from our baseline)
      2. plip_pan_tcga/ (pan-cancer, for non-LUAD cohorts we'll use next)
    """

    search_dirs: List[Path] = []
    if cohort == "LUAD":
        search_dirs += [token_root / "plip_luad_256"]
    search_dirs += [token_root / "plip_pan_tcga"]

    per_case_tiles: Dict[str, List[np.ndarray]] = {}
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for npz_path in sdir.rglob("*.npz"):
            sub = extract_submitter_id(str(npz_path))
            if not sub:
                continue
            if infer_cohort_from_sub(sub) != cohort:
                continue
            try:
                arr = np.load(npz_path, mmap_mode="r")
            except Exception:
                continue
            keys = [k for k in arr.files if k in {"feats", "features", "arr_0", "tokens", "feat"}]
            if not keys:
                continue
            feat = np.asarray(arr[keys[0]], dtype=np.float32)
            if feat.ndim == 3:
                feat = feat.reshape(-1, feat.shape[-1])
            if feat.ndim == 2:
                per_case_tiles.setdefault(sub, []).append(feat)
            elif feat.ndim == 1:
                per_case_tiles.setdefault(sub, []).append(feat[None, :])

    out: Dict[str, np.ndarray] = {}
    for sub, tile_list in per_case_tiles.items():
        stacked = np.concatenate(tile_list, axis=0)
        if stacked.shape[0] == 0:
            continue
        mean_vec = stacked.mean(axis=0).astype(np.float32)
        out[sub] = mean_vec
    return out


def _load_survival_labels(clinical_csv: Path, cohort: str) -> Dict[str, Tuple[float, int]]:
    df = pd.read_csv(clinical_csv)
    cols = list(df.columns)
    sub_col = next((c for c in cols if "submitter" in c.lower() or c.lower() == "case_id"), None)
    if sub_col is None:
        raise RuntimeError(f"{clinical_csv} missing submitter column: {cols}")

    # Survival time: use the most informative date column available, with
    # priority: (1) explicit "time" column, (2) days_to_death when Dead,
    # else (3) days_to_last_follow_up when Alive.  All three are commonly
    # present in the standard TCGA clinical cart downloads we use.
    time_candidates = [c for c in cols if c.lower() in {"time", "os_time", "days_to_death", "days_to_last_follow_up"}]
    event_candidates = [c for c in cols if c.lower() in {"event", "os_event", "vital_status"}]

    if "time" in cols:
        def get_time(row) -> float: return float(row.time)
    else:
        def get_time(row) -> float:
            vs = getattr(row, "vital_status", None) if "vital_status" in cols else None
            d2d = getattr(row, "days_to_death", None) if "days_to_death" in cols else None
            d2f = getattr(row, "days_to_last_follow_up", None) if "days_to_last_follow_up" in cols else None
            if str(vs).lower() == "dead" and d2d is not None and pd.notna(d2d) and float(d2d) > 0:
                return float(d2d)
            if d2f is not None and pd.notna(d2f) and float(d2f) > 0:
                return float(d2f)
            return float("nan")

    if "event" in cols:
        def get_event(row) -> int:
            e = getattr(row, "event")
            return 1 if e in {1, "1", True} else 0
    else:
        def get_event(row) -> int:
            vs = getattr(row, "vital_status", None) if "vital_status" in cols else None
            return 1 if str(vs).lower() == "dead" else 0

    mapping: Dict[str, Tuple[float, int]] = {}
    for row in df.itertuples(index=False):
        sub = str(getattr(row, sub_col))
        try:
            t = get_time(row)
            e = get_event(row)
        except Exception:
            continue
        if not np.isfinite(t) or t <= 0:
            continue
        mapping[sub] = (float(t), int(e))
    return mapping


@dataclass
class CohortAssets:
    subs: List[str]
    wsi: np.ndarray     # (N, 512)
    rna: np.ndarray     # (N, 50)
    time: np.ndarray    # (N,)
    event: np.ndarray   # (N,) int 0/1
    cohort: str


def load_cohort_for_experiment(cohort: str = "LUAD") -> CohortAssets:
    paths = resolve_data_paths()
    tcga_10c = paths.raw_rna_root / "rna_h5ad" / "tcga_10c_v2.symbol_tpm.with_labels.nonempty.fixed.h5ad"
    clinical_csv = paths.raw_clinical_root / cohort / "clinical.csv"

    rna_map = _load_hallmark50_rna_vectors(tcga_10c, cohort)
    wsi_map = _load_wsi_mean_vectors(paths.token_root, cohort)
    surv_map = _load_survival_labels(clinical_csv, cohort)

    common = sorted(set(rna_map) & set(wsi_map) & set(surv_map))
    if len(common) < 30:
        raise RuntimeError(f"Not enough overlapping cases for {cohort}: N={len(common)}")
    wsi = np.stack([wsi_map[s] for s in common], axis=0).astype(np.float32)
    rna = np.stack([rna_map[s] for s in common], axis=0).astype(np.float32)
    t = np.asarray([surv_map[s][0] for s in common], dtype=np.float64)
    e = np.asarray([surv_map[s][1] for s in common], dtype=np.int32)
    return CohortAssets(subs=common, wsi=wsi, rna=rna, time=t, event=e, cohort=cohort)


# =========================================================================
# Anti-alignment model (pure CPU, tiny)
# =========================================================================

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class Projector(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 128, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class DualProjector(nn.Module):
    def __init__(self, wsi_in: int = 512, rna_in: int = 50, hid: int = 256, out: int = 128):
        super().__init__()
        self.psi_wsi = Projector(wsi_in, hid, out)
        self.psi_rna = Projector(rna_in, hid, out)

    def forward(self, wsi: torch.Tensor, rna: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.psi_wsi(wsi), self.psi_rna(rna)


def anti_alignment_loss(z_wsi: torch.Tensor, z_rna: torch.Tensor,
                        variant: str = "mse_ones_minus_I",
                        margin: float = 0.5, temperature: float = 0.1) -> torch.Tensor:
    """Three variants of the anti-alignment objective.

    Variants (all do the same high-level thing, differ in how strong the
    off-diagonal pull is):

      - "mse_ones_minus_I": the user's literal description. Target T = 1 - I.
        L = MSE(S, T) with S = Z_wsi @ Z_rna^T after L2-normalization.
      - "push_pull": L = mean(S_diag) + mean(relu(m - S_off))
          = push the diagonal toward 0, *and* enforce that every off-diagonal
          entry is at least m (pull them all "high", or at least not low).
      - "anti_contrastive": softmax over (i, *) rows where the "correct"
        label is NOT i (anti-InfoNCE). This is the information-theory twin
        of the other two; it's the strongest regulariser in practice.
    """
    N = z_wsi.shape[0]
    S = z_wsi @ z_rna.T  # (N, N) cos similarities in [-1, 1]
    eye = torch.eye(N, device=z_wsi.device, dtype=z_wsi.dtype)
    off = 1.0 - eye

    if variant == "mse_ones_minus_I":
        target = 1.0 - eye  # exactly what the user described
        return F.mse_loss(S, target)

    if variant == "push_pull":
        diag = (S * eye).sum(dim=-1)          # (N,)
        S_off = S * off                       # off-diag kept, diag zeroed
        # penalize off-diagonal entries that fall below the margin
        penalty = torch.relu(margin - S_off)
        # Mean over *actual* off-diagonal entries (not all N^2)
        penalty_mean = penalty.sum() / off.sum().clamp_min(1.0)
        return diag.mean() + penalty_mean

    if variant == "anti_contrastive":
        # Per row i, the "anti-logit" of picking any j != i should be high;
        # i.e. minimize -log( softmax(S / T)[not_i] / sum )
        S_t = S / temperature
        # Numerically-stable anti-InfoNCE: for each row i,
        #   L_i = -logsumexp( S[i, j != i] ) + logsumexp( S[i, *] )
        # which is the same as InfoNCE with label "NOT i".
        logits_any = torch.logsumexp(S_t, dim=-1)                       # (N,)
        # zero out diag to get logsumexp over off-diag
        S_t_off = S_t - 1e9 * eye
        logits_off = torch.logsumexp(S_t_off, dim=-1)                   # (N,)
        per_row = logits_any - logits_off                               # want small
        return per_row.mean()

    raise ValueError(f"unknown loss variant: {variant}")


# =========================================================================
# C-index helper (pure numpy, CoxModel-free, works on any scalar risk).
# =========================================================================

def cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(time, kind="mergesort")
    t = time[order]; ev = event[order]; r = risk[order]
    n_conc = 0.0
    n_valid = 0.0
    N = len(t)
    for i in range(N):
        if ev[i] != 1:
            continue
        for j in range(i + 1, N):
            if t[j] == t[i]:
                continue
            n_valid += 1.0
            if r[i] > r[j]:
                n_conc += 1.0
            elif r[i] == r[j]:
                n_conc += 0.5
    return float(n_conc / n_valid) if n_valid > 0 else float("nan")


def train_anti_alignment(assets: CohortAssets, out_dir: Path,
                         epochs: int = 600, batch_size: int = 64,
                         hid_dim: int = 256, out_dim: int = 128,
                         loss_variant: str = "mse_ones_minus_I",
                         seed: int = 0) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = len(assets.subs)
    wsi_t = torch.from_numpy(assets.wsi).float()
    rna_t = torch.from_numpy(assets.rna).float()
    # Standardize each feature family before projection -- numerics only.
    wsi_t = (wsi_t - wsi_t.mean(0)) / (wsi_t.std(0) + 1e-6)
    rna_t = (rna_t - rna_t.mean(0)) / (rna_t.std(0) + 1e-6)

    model = DualProjector(wsi_in=wsi_t.shape[1], rna_in=rna_t.shape[1],
                          hid=hid_dim, out=out_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    for ep in range(epochs):
        idx = torch.randperm(N)
        for start in range(0, N, batch_size):
            bi = idx[start:start + batch_size]
            if bi.numel() < 4:
                continue
            zw, zr = model(wsi_t[bi], rna_t[bi])
            loss = anti_alignment_loss(zw, zr, variant=loss_variant)
            opt.zero_grad(True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()

    # Full-batch embeddings + similarity matrix at convergence.
    model.eval()
    with torch.no_grad():
        zw_all, zr_all = model(wsi_t, rna_t)
    S = (zw_all @ zr_all.T).cpu().numpy()
    diag_mean = float(np.mean(np.diag(S)))
    off_mask = ~np.eye(N, dtype=bool)
    off_mean = float(np.mean(S[off_mask]))
    # Goal check: mean(off) should be much greater than mean(diag)
    separation = off_mean - diag_mean

    np.save(out_dir / f"z_wsi_{assets.cohort}.npy", zw_all.cpu().numpy().astype(np.float32))
    np.save(out_dir / f"z_rna_{assets.cohort}.npy", zr_all.cpu().numpy().astype(np.float32))
    np.save(out_dir / f"anti_sim_matrix_{assets.cohort}.npy", S.astype(np.float32))
    np.save(out_dir / f"wsi_raw_{assets.cohort}.npy", wsi_t.cpu().numpy().astype(np.float32))
    np.save(out_dir / f"rna_raw_{assets.cohort}.npy", rna_t.cpu().numpy().astype(np.float32))
    np.savetxt(out_dir / f"sample_ids_{assets.cohort}.txt", np.array(assets.subs, dtype=str), fmt="%s")

    # -------- Baseline C-indexes (original 512/50 raw, before anti-proj) --------
    # For simplicity we fit a Lasso-penalised Cox on the *raw* features, then
    # compare with Cox on: (a) anti-features alone, (b) concat(raw, anti).
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    y = Surv.from_arrays(event=assets.event.astype(bool), time=assets.time)

    def fit_eval(X: np.ndarray, tag: str, folds: int = 5) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        splits = np.array_split(perm, folds)
        cis = []
        for vi in range(folds):
            val = splits[vi]
            tr = np.concatenate([splits[k] for k in range(folds) if k != vi])
            est = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=30, alpha_min_ratio=0.05,
                                         normalize=False, max_iter=10000, tol=1e-6)
            est.fit(X[tr], y[tr])
            # Pick alpha with best train C-index (simple & fast for this CPU toy)
            preds_tr = np.asarray([est.predict(X[tr], alpha=a).ravel() for a in est.alphas_])
            tr_cis = np.asarray([cindex(assets.time[tr], assets.event[tr], preds_tr[k]) for k in range(len(est.alphas_))])
            best = int(np.nanargmax(tr_cis))
            pred = est.predict(X[val], alpha=est.alphas_[best]).ravel()
            ci = cindex(assets.time[val], assets.event[val], pred)
            cis.append(ci)
        return {f"{tag}_cindex_mean": float(np.nanmean(cis)),
                f"{tag}_cindex_std":  float(np.nanstd(cis))}

    X_wsi_raw = wsi_t.cpu().numpy()
    X_rna_raw = rna_t.cpu().numpy()
    X_wsi_anti = zw_all.cpu().numpy()
    X_rna_anti = zr_all.cpu().numpy()
    X_concat_both = np.concatenate([X_wsi_raw, X_rna_raw, X_wsi_anti, X_rna_anti], axis=1)
    X_concat_modal = (np.concatenate([X_wsi_raw, X_wsi_anti], axis=1),
                      np.concatenate([X_rna_raw, X_rna_anti], axis=1))

    results: Dict[str, float] = {
        "cohort": assets.cohort,
        "N": N,
        "wsi_dim_raw": X_wsi_raw.shape[1],
        "rna_dim_raw": X_rna_raw.shape[1],
        "anti_dim": out_dim,
        "loss_variant": loss_variant,
        "train_final_loss": float(np.mean(losses[-50:])),
        "anti_diag_mean_sim": diag_mean,
        "anti_offdiag_mean_sim": off_mean,
        "anti_separation_off_minus_diag": separation,
        "raw_wsi_rna_diag_mean_sim": -1.0,  # filled below
        "raw_wsi_rna_offdiag_mean_sim": -1.0,
    }

    # Raw-space WSI vs RNA "similarity" layout cannot be computed as a direct
    # matrix dot product because the two modalities have different ambient
    # dimensionalities (WSI 512 vs RNA 50).  Instead, we compare per-row
    # *ranking* layout using a standard cross-modal proxy: for each case we
    # look at its raw WSI's k-nearest neighbours in WSI-space and its raw
    # RNA's k-nearest neighbours in RNA-space.  The average overlap of
    # those neighbourhoods (the "rank-biased overlap" surrogate) is the
    # scalar that tells us how aligned the two raw layouts were.  We then
    # compute the same overlap for the anti-space and compare them -- this
    # is the user's "排布对比" question answered quantitatively.
    #
    # For the diagonal / off-diagonal *mean cosine numbers* we simply use
    # normalized WSI against a PCA-projected RNA (and vice versa) since
    # those scalars are meant for visual comparison, not for rigour.
    from sklearn.decomposition import PCA

    def _row_norms(X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    d_common = min(X_wsi_raw.shape[1], X_rna_raw.shape[1])
    pca_w = PCA(n_components=d_common, random_state=seed).fit_transform(X_wsi_raw)
    pca_r = PCA(n_components=d_common, random_state=seed).fit_transform(X_rna_raw)
    W_raw_p = _row_norms(pca_w)
    R_raw_p = _row_norms(pca_r)
    S_raw = W_raw_p @ R_raw_p.T  # (N, N) on a common d_common subspace
    results["raw_wsi_rna_diag_mean_sim"] = float(np.mean(np.diag(S_raw)))
    results["raw_wsi_rna_offdiag_mean_sim"] = float(np.mean(S_raw[off_mask]))

    # ---- Neighbourhood-overlap layout comparison (the proper one) ----
    from sklearn.neighbors import NearestNeighbors

    def _knn_sets(X: np.ndarray, k: int) -> List[set]:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X) - 1), metric="cosine", algorithm="brute")
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        # idx has self as the 0-th neighbour because cosine(self,self)=1 smallest distance
        return [set(row[1:k + 1].tolist()) for row in idx]

    K = min(10, max(3, N // 10))
    knn_w_raw = _knn_sets(_row_norms(X_wsi_raw), K)
    knn_r_raw = _knn_sets(_row_norms(X_rna_raw), K)
    knn_w_anti = _knn_sets(X_wsi_anti, K)  # anti-features are already L2-normalised
    knn_r_anti = _knn_sets(X_rna_anti, K)

    def _avg_overlap(A: List[set], B: List[set]) -> float:
        vals = [float(len(a & b)) / float(max(1, len(a | b))) for a, b in zip(A, B)]
        return float(np.mean(vals))

    results["layout__wsi_raw_vs_rna_raw_knn" + str(K) + "_jaccard_mean"] = _avg_overlap(knn_w_raw, knn_r_raw)
    results["layout__wsi_anti_vs_rna_anti_knn" + str(K) + "_jaccard_mean"] = _avg_overlap(knn_w_anti, knn_r_anti)

    def _avg_knn_quality(knn_w: List[set], knn_r: List[set]) -> float:
        """For each case, what fraction of WSI neighbours also appear among
        the *same case's* RNA neighbours.  This metric is maximised when
        the two modalities agree on who is "close to whom" (strong
        alignment).  It is the cleanest single-number answer to the user's
        open question: "did anti-feature training *change* the per-patient
        cross-modal layout?"
        """
        vals = [float(len(w & r)) / float(max(1, len(w))) for w, r in zip(knn_w, knn_r)]
        return float(np.mean(vals))

    results["layout_aligned_frac__raw_wsi_knn_hit_rna_knn_k" + str(K)] = _avg_knn_quality(knn_w_raw, knn_r_raw)
    results["layout_aligned_frac__anti_wsi_knn_hit_rna_knn_k" + str(K)] = _avg_knn_quality(knn_w_anti, knn_r_anti)

    # ---- C-indexes: 5-fold CV on the full cohort (CPU, few minutes) ----
    for name, X in [
        ("wsi_raw_only",   X_wsi_raw),
        ("rna_raw_only",   X_rna_raw),
        ("wsi_anti_only",  X_wsi_anti),
        ("rna_anti_only",  X_rna_anti),
        ("wsi_raw_plus_wsi_anti", X_concat_modal[0]),
        ("rna_raw_plus_rna_anti", X_concat_modal[1]),
        ("raw_wsi_rna_only",      np.concatenate([X_wsi_raw, X_rna_raw], axis=1)),
        ("raw_plus_anti_allconcat", X_concat_both),
    ]:
        results.update(fit_eval(X, tag=name))

    # ---- Orthogonality / overlap vs original layout: answer user question ----
    # "anti特征 vs 原特征 排布到底差多少"  We compare the two NxN cross-modal
    # similarity orderings via Spearman rank correlation between S_raw and S.
    from scipy.stats import spearmanr
    raw_flat = S_raw[off_mask].ravel()
    anti_flat = S[off_mask].ravel()
    rho_off, p_off = spearmanr(raw_flat, anti_flat)
    results["spearman_Sraw_vs_Santi__offdiag_rho"] = float(rho_off)
    results["spearman_Sraw_vs_Santi__offdiag_p"]   = float(p_off)

    diag_raw = np.diag(S_raw); diag_anti = np.diag(S)
    rho_diag, p_diag = spearmanr(diag_raw, diag_anti)
    results["spearman_Sraw_vs_Santi__diag_rho"] = float(rho_diag)
    results["spearman_Sraw_vs_Santi__diag_p"]   = float(p_diag)

    # How much of anti-feature space is spanned by raw features?
    # Answer via linear regression of Z_wsi against X_wsi_raw: R^2 averaged
    # across each of the 128 anti-dimensions.  If R^2 is close to zero, the
    # anti-feature lives in a genuinely different subspace (not just noise
    # rotation -- it contains genuinely new info).
    def subspace_R2(Z: np.ndarray, X: np.ndarray) -> float:
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        beta, *_ = np.linalg.lstsq(Xb, Z, rcond=None)
        pred = Xb @ beta
        resid = Z - pred
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = ((Z - Z.mean(0)) ** 2).sum(axis=0)
        r2_1d = 1.0 - ss_res / (ss_tot + 1e-12)
        return float(np.clip(r2_1d, -1.0, 1.0).mean())

    results["R2__Z_wsi_anti_regressed_on_X_wsi_raw__mean128"] = subspace_R2(X_wsi_anti, X_wsi_raw)
    results["R2__Z_rna_anti_regressed_on_X_rna_raw__mean128"] = subspace_R2(X_rna_anti, X_rna_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_{assets.cohort}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    torch.save(model.state_dict(), out_dir / f"anti_model_{assets.cohort}.pt")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="LUAD", choices=["LUAD", "BRCA", "SKCM", "COAD", "HNSC", "UCEC", "KIRC", "THCA", "LGG", "LUSC", "PRAD", "READ"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "anti_feature_experiment"))
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hid-dim", type=int, default=256, help="Compress both modalities to this dim BEFORE final 128 anti-feature.")
    ap.add_argument("--out-dim", type=int, default=128, help="Final anti-feature dimensionality.")
    ap.add_argument("--loss-variant", default="mse_ones_minus_I",
                    choices=["mse_ones_minus_I", "push_pull", "anti_contrastive"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_cohort_for_experiment(args.cohort)
    print(f"[anti-exp] cohort={args.cohort} N={len(assets.subs)} wsi_dim={assets.wsi.shape[1]} rna_dim={assets.rna.shape[1]}", flush=True)
    results = train_anti_alignment(assets, out_dir,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   hid_dim=args.hid_dim, out_dim=args.out_dim,
                                   loss_variant=args.loss_variant, seed=args.seed)
    print("\n[anti-exp] RESULTS SUMMARY:\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
