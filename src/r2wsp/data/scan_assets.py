from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .paths import DataPaths


TCGA_COHORTS = (
    "BLCA",
    "BRCA",
    "COAD",
    "ESCA",
    "HNSC",
    "KIRC",
    "LGG",
    "LUAD",
    "LUSC",
    "PAAD",
    "PRAD",
    "READ",
    "SKCM",
    "TGCT",
    "THCA",
    "UCEC",
)

TCGA_SUBMITTER_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def infer_cohort(text: str) -> str | None:
    upper = text.upper()
    for cohort in TCGA_COHORTS:
        if cohort in upper:
            return cohort
    return None


def extract_submitter_id(text: str) -> str | None:
    m = TCGA_SUBMITTER_RE.search(text.upper())
    if m is None:
        return None
    return m.group(0)


@dataclass(frozen=True)
class SVSAsset:
    path: Path
    file_name: str
    slide_id: str
    submitter_id: str | None
    cohort: str | None


@dataclass(frozen=True)
class TokenAsset:
    path: Path
    file_name: str
    token_id: str
    source_name: str
    feature_kind: str
    submitter_id: str | None
    cohort: str | None


@dataclass(frozen=True)
class RNAAsset:
    path: Path
    file_name: str
    source_name: str
    cohort: str | None
    file_type: str


@dataclass(frozen=True)
class ClinicalAsset:
    path: Path
    cohort: str


@dataclass(frozen=True)
class AssetInventory:
    svs: list[SVSAsset]
    tokens: list[TokenAsset]
    rna: list[RNAAsset]
    clinical: list[ClinicalAsset]

    def summary(self) -> dict[str, object]:
        return {
            "svs_count": len(self.svs),
            "token_count": len(self.tokens),
            "rna_count": len(self.rna),
            "clinical_count": len(self.clinical),
            "svs_by_cohort": _count_by(self.svs, lambda x: x.cohort or "UNKNOWN"),
            "tokens_by_source": _count_by(self.tokens, lambda x: x.source_name),
            "rna_by_source": _count_by(self.rna, lambda x: x.source_name),
            "clinical_by_cohort": _count_by(self.clinical, lambda x: x.cohort),
        }


def _count_by(items: list[object], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def scan_svs_assets(root: str | Path) -> list[SVSAsset]:
    base = Path(root)
    assets: list[SVSAsset] = []
    if not base.exists():
        return assets

    for path in sorted(base.rglob("*.svs")):
        text = str(path)
        assets.append(
            SVSAsset(
                path=path,
                file_name=path.name,
                slide_id=path.stem,
                submitter_id=extract_submitter_id(text),
                cohort=infer_cohort(text),
            )
        )
    return assets


def _append_token_asset(
    assets: list[TokenAsset],
    *,
    path: Path,
    source_name: str,
    feature_kind: str,
    cohort_hint: str | None = None,
) -> None:
    text = str(path)
    assets.append(
        TokenAsset(
            path=path,
            file_name=path.name,
            token_id=path.stem,
            source_name=source_name,
            feature_kind=feature_kind,
            submitter_id=extract_submitter_id(text),
            cohort=infer_cohort(text) or infer_cohort(source_name) or cohort_hint,
        )
    )


def scan_token_assets(root: str | Path, extra_wsi_feature_root: str | Path | None = None) -> list[TokenAsset]:
    base = Path(root)
    assets: list[TokenAsset] = []
    if base.exists():
        for source_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for path in sorted(source_dir.rglob("*.npz")):
                _append_token_asset(
                    assets,
                    path=path,
                    source_name=source_dir.name,
                    feature_kind="npz_tokens",
                )

    if extra_wsi_feature_root is not None:
        feature_base = Path(extra_wsi_feature_root)
        if feature_base.exists():
            for feats_dir in sorted(
                p for p in feature_base.rglob("*") if p.is_dir() and p.name in {"feats_h5", "feats_pt"}
            ):
                source_name = feats_dir.parent.name
                feature_kind = "h5_features" if feats_dir.name == "feats_h5" else "pt_features"
                suffix = "*.h5" if feature_kind == "h5_features" else "*.pt"
                for path in sorted(feats_dir.rglob(suffix)):
                    _append_token_asset(
                        assets,
                        path=path,
                        source_name=source_name,
                        feature_kind=feature_kind,
                    )
    return assets


def scan_rna_assets(root: str | Path) -> list[RNAAsset]:
    base = Path(root)
    assets: list[RNAAsset] = []
    if not base.exists():
        return assets

    for source_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for suffix in ("*.tsv", "*.h5ad", "*.csv"):
            for path in sorted(source_dir.rglob(suffix)):
                if path.name.startswith("."):
                    continue
                assets.append(
                    RNAAsset(
                        path=path,
                        file_name=path.name,
                        source_name=source_dir.name,
                        cohort=infer_cohort(path.name) or infer_cohort(source_dir.name),
                        file_type=path.suffix.lower().lstrip("."),
                    )
                )
    return assets


def scan_clinical_assets(root: str | Path) -> list[ClinicalAsset]:
    base = Path(root)
    assets: list[ClinicalAsset] = []
    if not base.exists():
        return assets

    for cohort_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        clinical_csv = cohort_dir / "clinical.csv"
        if clinical_csv.exists():
            assets.append(ClinicalAsset(path=clinical_csv, cohort=cohort_dir.name.upper()))
    return assets


def scan_all_assets(paths: DataPaths) -> AssetInventory:
    return AssetInventory(
        svs=scan_svs_assets(paths.raw_svs_root),
        tokens=scan_token_assets(paths.token_root, paths.data_root / "wsi_features"),
        rna=scan_rna_assets(paths.raw_rna_root),
        clinical=scan_clinical_assets(paths.raw_clinical_root),
    )
