from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .scan_assets import AssetInventory, RNAAsset, SVSAsset, TokenAsset


@dataclass(frozen=True)
class IndexRow:
    cohort: str | None
    case_id: str | None
    slide_id: str
    svs_path: Path | None
    wsi_feature_path: Path
    wsi_feature_source: str
    wsi_feature_kind: str
    rna_path: Path | None
    clinical_path: Path | None

    @property
    def token_path(self) -> Path:
        return self.wsi_feature_path

    @property
    def token_source(self) -> str:
        return self.wsi_feature_source


def _pick_rna_per_cohort(rna_assets: list[RNAAsset]) -> dict[str, Path]:
    by_cohort: dict[str, list[RNAAsset]] = {}
    for a in rna_assets:
        if a.cohort is None:
            continue
        by_cohort.setdefault(a.cohort.upper(), []).append(a)

    chosen: dict[str, Path] = {}
    for cohort, items in by_cohort.items():
        tsv = [x for x in items if x.file_type == "tsv"]
        if tsv:
            prefer = [x for x in tsv if "tpm_tsv" in x.source_name.lower()]
            pick = prefer[0] if prefer else tsv[0]
            chosen[cohort] = pick.path
            continue
        chosen[cohort] = items[0].path
    return chosen


def _svs_maps(svs_assets: list[SVSAsset]) -> tuple[dict[str, SVSAsset], dict[str, list[SVSAsset]]]:
    by_slide: dict[str, SVSAsset] = {}
    by_case: dict[str, list[SVSAsset]] = {}
    for s in svs_assets:
        by_slide[s.slide_id] = s
        if s.submitter_id is not None:
            by_case.setdefault(s.submitter_id, []).append(s)
    return by_slide, by_case


def _choose_svs_for_token(token: TokenAsset, *, by_slide: dict[str, SVSAsset], by_case: dict[str, list[SVSAsset]]) -> SVSAsset | None:
    if token.token_id in by_slide:
        return by_slide[token.token_id]
    if token.submitter_id is None or token.submitter_id not in by_case:
        return None

    candidates = by_case[token.submitter_id]
    if not candidates:
        return None

    best = None
    best_score = None
    tid = token.token_id
    for s in candidates:
        sid = s.slide_id
        score = 0
        if tid == sid:
            score += 100
        if tid in sid:
            score += 10
        if sid in tid:
            score += 10
        if token.file_name.startswith(sid):
            score += 5
        if best is None or score > int(best_score):
            best = s
            best_score = score
    return best


def build_index(inventory: AssetInventory) -> list[IndexRow]:
    clinical_by_cohort = {c.cohort.upper(): c.path for c in inventory.clinical}
    rna_by_cohort = _pick_rna_per_cohort(inventory.rna)
    by_slide, by_case = _svs_maps(inventory.svs)

    rows: list[IndexRow] = []
    for t in inventory.tokens:
        svs = _choose_svs_for_token(t, by_slide=by_slide, by_case=by_case)
        inferred_cohort = t.cohort or (svs.cohort if svs is not None else None)
        cohort = (inferred_cohort or "").upper() or None
        case_id = t.submitter_id or (svs.submitter_id if svs is not None else None)
        slide_id = t.token_id
        rna_path = rna_by_cohort.get(cohort) if cohort is not None else None
        clinical_path = clinical_by_cohort.get(cohort) if cohort is not None else None
        rows.append(
            IndexRow(
                cohort=cohort,
                case_id=case_id,
                slide_id=slide_id,
                svs_path=svs.path if svs is not None else None,
                wsi_feature_path=t.path,
                wsi_feature_source=t.source_name,
                wsi_feature_kind=t.feature_kind,
                rna_path=rna_path,
                clinical_path=clinical_path,
            )
        )
    return rows


def summarize_index(rows: list[IndexRow]) -> dict[str, object]:
    total = len(rows)
    with_case = sum(1 for r in rows if r.case_id is not None)
    with_svs = sum(1 for r in rows if r.svs_path is not None)
    with_rna = sum(1 for r in rows if r.rna_path is not None)
    with_clinical = sum(1 for r in rows if r.clinical_path is not None)
    by_cohort: dict[str, int] = {}
    for r in rows:
        key = r.cohort or "UNKNOWN"
        by_cohort[key] = by_cohort.get(key, 0) + 1
    by_cohort = dict(sorted(by_cohort.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "rows": total,
        "with_case_id": with_case,
        "with_svs": with_svs,
        "with_rna": with_rna,
        "with_clinical": with_clinical,
        "by_cohort": by_cohort,
    }
