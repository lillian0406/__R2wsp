from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.tcga_dataset import TCGAMultimodalDataset
from r2wsp.models.retrieval import R2wspRetrieval
from r2wsp.train.collate import collate_tiles_and_rna


def _enrich_rows_with_cohort_from_svs5063(
    rows,
    svs5063_csv: Path = _ROOT / "outputs" / "_disk_cohort_inventory" / "svs_5063_cohort_audit_TCGA_OFFICIAL_FINAL.csv",
):
    """Patch IndexRow.cohort / rna_path / clinical_path via GDC 100% cohort audit (9-col noheader).

    scan_assets infers cohort from source directory names, which misses UNI1024 feats_h5/
    (single mixed cohort dir under ``extracted_mag20x_patch256_fp/uni1024/feats_h5``).
    svs5063 col 3 = case_id_submitter TCGA-XX-XXXX (3-seg), col 5 = GDC cohort (LUSC/LUAD/BRCA/BLCA/...).
    Rows whose cohort is already set (e.g. PLIP LUAD source dirs) are untouched.
    Rows with enriched cohort also receive rna_path/clinical_path fallbacks via
    the per-cohort dictionaries returned by ``resolve_data_paths + scan_all_assets``,
    using the same selection rules as ``build_index`` (TSV preferred for rna, exact cohort match).
    """
    if not svs5063_csv.is_file():
        return rows
    case_short_to_cohort: dict[str, str] = {}
    with svs5063_csv.open(newline="") as fh:
        for rec in csv.reader(fh):
            if len(rec) < 6:
                continue
            case_raw = str(rec[3]).strip().upper()
            coh = str(rec[5]).strip()
            if not case_raw.startswith("TCGA-") or not coh:
                continue
            segs = case_raw.split("-")
            if len(segs) < 3:
                continue
            case_short_to_cohort["-".join(segs[:3])] = coh.upper()
    if not case_short_to_cohort:
        return rows
    # build simple per-cohort rna/clinical dicts from existing rows (already populated by build_index for LUAD via PLIP)
    rna_by_cohort: dict[str, Path] = {}
    clinical_by_cohort: dict[str, Path] = {}
    # HIGHEST PRIORITY: pre-exported symbol TPM from tcga_10c h5ad (gene names == HGNC symbol, hallmark 99.9% intersect)
    tpm_export_dir = _ROOT / "outputs" / "_disk_cohort_inventory"
    symbol_tpm_cases: dict[str, set[str]] = {}
    if tpm_export_dir.is_dir():
        for tsv in sorted(tpm_export_dir.glob("TCGA_*_symbol_tpm.tsv")):
            try:
                name = tsv.stem[len("TCGA_"):-len("_symbol_tpm")].upper()
            except Exception:
                name = None
            if not name:
                continue
            rna_by_cohort[name] = tsv.resolve()
            try:
                with open(tsv, "r") as f:
                    header = f.readline()
                cols = [c.strip().strip('"') for c in header.split("\t")]
                case_cols = set()
                for c in cols:
                    segs = c.upper().split("-")
                    if len(segs) >= 3 and segs[0] == "TCGA":
                        case_cols.add("-".join(segs[:3]))
                symbol_tpm_cases[name] = case_cols
            except Exception:
                pass
    for r in rows:
        if r.cohort is None:
            continue
        if r.rna_path is not None and r.cohort not in rna_by_cohort:
            rna_by_cohort[r.cohort] = r.rna_path
        if r.clinical_path is not None and r.cohort not in clinical_by_cohort:
            clinical_by_cohort[r.cohort] = r.clinical_path
    # fallback: scan raw_rna/tpm_tsv/<cohort>/ (may contain ENSG IDs; caller should pass rna_gene_id_to_symbol when using this branch)
    for cohort_dir in sorted((_ROOT / "data" / "raw_rna" / "tpm_tsv").iterdir()):
        if not cohort_dir.is_dir():
            continue
        name = cohort_dir.name.upper()
        if name in rna_by_cohort:
            continue
        tsvs = sorted(cohort_dir.glob("*.tsv")) + sorted(cohort_dir.glob("*.csv"))
        if tsvs:
            rna_by_cohort[name] = tsvs[0].resolve()
    clinical_root = _ROOT / "data" / "raw_clinical"
    for cohort_dir in sorted(clinical_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        name = cohort_dir.name.upper()
        if name in clinical_by_cohort:
            continue
        p = cohort_dir / "clinical.csv"
        if p.is_file():
            clinical_by_cohort[name] = p.resolve()
    enriched = []
    for r in rows:
        if r.cohort is not None and r.rna_path is not None and r.clinical_path is not None:
            enriched.append(r)
            continue
        case_short = None
        if r.case_id is not None:
            segs = str(r.case_id).strip().upper().split("-")
            if len(segs) >= 3:
                case_short = "-".join(segs[:3])
        if case_short is None:
            enriched.append(r)
            continue
        coh = case_short_to_cohort.get(case_short, r.cohort if r.cohort is not None else None)
        if coh is None:
            enriched.append(r)
            continue
        new_cohort = r.cohort if r.cohort is not None else coh
        new_rna = r.rna_path if r.rna_path is not None else rna_by_cohort.get(new_cohort)
        if (
            new_rna is not None
            and new_cohort in symbol_tpm_cases
            and case_short not in symbol_tpm_cases[new_cohort]
        ):
            new_rna = None
        new_clinical = (
            r.clinical_path if r.clinical_path is not None else clinical_by_cohort.get(new_cohort)
        )
        if (
            new_cohort == r.cohort
            and new_rna == r.rna_path
            and new_clinical == r.clinical_path
        ):
            enriched.append(r)
            continue
        enriched.append(
            replace(
                r,
                cohort=new_cohort,
                rna_path=new_rna,
                clinical_path=new_clinical,
            )
        )
    return enriched


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--token_source", default=None)
    p.add_argument("--max_tiles", type=int, default=256)
    p.add_argument("--rna_mode", default="vec", choices=["vec", "omics"])
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_rna_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--out_dir", default="outputs/retrieval_run")
    p.add_argument("--tile_dim", type=int, default=512, help="WSI tile token embedding dim (512 PLIP / 1024 UNI1024 / 768 CtransPath Lunit)")
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    paths.validate()

    inv = scan_all_assets(paths)
    rows = build_index(inv)
    rows = _enrich_rows_with_cohort_from_svs5063(rows)
    # Optional --cohort CLI filter can be added later; default: train on all rows available

    gene_sets_csv = None
    if str(args.rna_mode) == "omics":
        gene_sets_csv = (
            Path(args.rna_gene_sets_csv).resolve()
            if args.rna_gene_sets_csv is not None
            else (paths.raw_rna_root / "metadata" / "hallmarks_signatures.csv").resolve()
        )
    # TCGAMultimodalDataset filters by rna_mode=omics strictly requires explicit gene_sets_csv
    if str(args.rna_mode) == "omics" and gene_sets_csv is None:
        raise RuntimeError("rna_mode=omics requires gene_sets_csv")

    ds_train = TCGAMultimodalDataset(
        rows,
        split="train",
        seed=int(args.seed),
        token_source=args.token_source,
        max_tiles=int(args.max_tiles),
        rna_mode=str(args.rna_mode),
        rna_gene_sets_csv=gene_sets_csv,
    )
    ds_val = TCGAMultimodalDataset(
        rows,
        split="val",
        seed=int(args.seed),
        token_source=args.token_source,
        max_tiles=int(args.max_tiles),
        rna_mode=str(args.rna_mode),
        rna_gene_sets_csv=gene_sets_csv,
    )

    device = torch.device(str(args.device))
    if str(args.rna_mode) == "omics":
        omic_sizes = ds_train.get_omics_spec().omic_sizes
        rna_encoder = "omics_mlp"
    else:
        omic_sizes = None
        rna_encoder = "mean"
    model = R2wspRetrieval(
        tile_dim=int(args.tile_dim),
        dim=int(args.dim),
        num_heads=int(args.heads),
        dropout=float(args.dropout),
        num_rna_tokens=int(args.num_rna_tokens),
        rna_encoder=rna_encoder,
        omic_sizes=omic_sizes,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(
        ds_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tiles_and_rna,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_tiles_and_rna,
    )

    def _run(loader: DataLoader, *, train: bool, max_batches: int) -> dict[str, float]:
        model.train(train)
        total = 0
        sum_loss = 0.0
        for step, batch in enumerate(loader):
            if int(max_batches) > 0 and int(step) >= int(max_batches):
                break
            tt = batch.tile_tokens.to(device)
            am = batch.tile_attn_mask.to(device)
            rv = batch.rna_vec.to(device) if batch.rna_vec is not None else None
            ro = [x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None
            out = model(tile_tokens=tt, tile_attn_mask=am, rna_vec=rv, rna_omics=ro)
            losses = model.compute_loss(out, temperature=float(args.temperature))
            loss = losses["loss"]
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            bsz = int(tt.shape[0])
            total += bsz
            sum_loss += float(loss.detach().cpu()) * bsz
        return {"loss": sum_loss / max(1, total), "n": total}

    metrics: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        tr = _run(train_loader, train=True, max_batches=int(args.max_train_batches))
        va = _run(val_loader, train=False, max_batches=int(args.max_val_batches))
        row = {"epoch": epoch, "train": tr, "val": va}
        metrics.append(row)
        (out_dir / "metrics.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        print(f"epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} train_n={tr['n']} val_n={va['n']}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "last.pt")


if __name__ == "__main__":
    main()
