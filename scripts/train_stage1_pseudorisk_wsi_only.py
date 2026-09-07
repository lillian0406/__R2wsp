from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets
from r2wsp.data.tcga_dataset import TCGAMultimodalDataset
from r2wsp.models.direct_survival import DirectWSIRNASurvival
from r2wsp.rna.gene_sets import load_gene_sets_csv
from r2wsp.rna.tokenizer import build_omics_spec
from r2wsp.train.collate import collate_tiles_and_rna


def set_seed(seed: int) -> None:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _strip_gene_version(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _split_train_val(case_ids: list[str], *, seed: int, val_frac: float) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(int(seed))
    ids = list(case_ids)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * float(val_frac)))
    n_val = max(1, min(n_val, len(ids) - 1)) if len(ids) > 1 else 0
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def _resolve_case_ids(split_dir: Path) -> tuple[list[str], list[str]]:
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    if "case_id" not in train_df.columns or "case_id" not in test_df.columns:
        raise ValueError("split csv must contain case_id")
    train_ids = sorted(set(train_df["case_id"].astype(str).tolist()))
    test_ids = sorted(set(test_df["case_id"].astype(str).tolist()))
    return train_ids, test_ids


def _load_pseudorisk_map(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "case_id" not in df.columns or "pseudo_risk" not in df.columns:
        raise ValueError("pseudo_risk csv must contain case_id and pseudo_risk")
    df["pseudo_risk"] = pd.to_numeric(df["pseudo_risk"], errors="coerce")
    df = df.dropna(subset=["pseudo_risk"])
    return {str(r.case_id): float(r.pseudo_risk) for r in df.itertuples(index=False)}


def _build_dataset_rows(
    all_rows,
    *,
    cohort: str,
    case_ids: set[str],
    wsi_feature_source: str,
    use_multi_slide: bool,
) -> list:
    rows = [
        r
        for r in all_rows
        if r.case_id is not None
        and str(r.case_id) in case_ids
        and str(r.wsi_feature_source) == str(wsi_feature_source)
    ]
    return rows


def _build_model(
    *,
    hidden_dim: int,
    dropout: float,
    rna_mode: str,
    rna_omic_sizes: list[int] | None,
    cross_modal_fusion: str,
    use_multi_slide: bool,
    multi_slide_mode: str,
) -> DirectWSIRNASurvival:
    return DirectWSIRNASurvival(
        tile_dim=1024,
        rna_dim=19962,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        model_variant="baseline",
        pool_method="attention",
        gate_enabled=None,
        use_multi_slide=bool(use_multi_slide),
        multi_slide_mode=str(multi_slide_mode),
        rna_mode=str(rna_mode),
        rna_omic_sizes=rna_omic_sizes,
        cross_modal_fusion=str(cross_modal_fusion),
        use_anti_injection=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, help="phase2_independent5/fold_k/eval_with_censored")
    p.add_argument("--cohort", required=True)
    p.add_argument("--pseudo_risk_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--max_tiles", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--use_multi_slide", action="store_true")
    p.add_argument("--multi_slide_mode", default="slide_attn_case_attn", choices=["slide_mean_case_attn", "slide_attn_case_attn"])
    p.add_argument("--multi_slide_tile_budget_mode", default="case_shared", choices=["per_slide", "case_shared"])
    p.add_argument("--wsi_feature_source", default="uni1024")
    p.add_argument("--rna_gene_sets_csv", default=None)
    p.add_argument("--gene_annotation_gtf", default=None)
    args = p.parse_args()

    set_seed(int(args.seed))
    split_dir = Path(args.split_dir).resolve()
    cohort = str(args.cohort).upper()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histories").mkdir(parents=True, exist_ok=True)

    train_ids, test_ids = _resolve_case_ids(split_dir)
    train_ids, val_ids = _split_train_val(train_ids, seed=int(args.seed), val_frac=float(args.val_frac))
    pseudo = _load_pseudorisk_map(Path(args.pseudo_risk_csv).resolve())

    train_ids = [c for c in train_ids if c in pseudo]
    val_ids = [c for c in val_ids if c in pseudo]
    test_ids = [c for c in test_ids if c in pseudo]
    if not train_ids or not val_ids:
        raise ValueError("empty train/val after pseudo_risk alignment")

    paths = resolve_data_paths()
    inv = scan_all_assets(paths)
    all_rows = build_index(inv)

    rna_tpm = (paths.raw_rna_root / "tpm_tsv" / f"{cohort.lower()}_tpm.tsv").resolve()
    clinical_csv = (paths.raw_clinical_root / cohort / "clinical.csv").resolve()
    if not rna_tpm.exists():
        raise FileNotFoundError(f"TPM TSV not found: {rna_tpm}")
    if not clinical_csv.exists():
        raise FileNotFoundError(f"clinical.csv not found: {clinical_csv}")

    keep_ids = set(train_ids) | set(val_ids) | set(test_ids)
    ds_rows = _build_dataset_rows(
        all_rows,
        cohort=cohort,
        case_ids=keep_ids,
        wsi_feature_source=str(args.wsi_feature_source),
        use_multi_slide=bool(args.use_multi_slide),
    )
    if not ds_rows:
        raise ValueError(
            f"no token rows matched by case_id for cohort={cohort} source={args.wsi_feature_source}. "
            f"Try checking whether uni1024 feats filenames contain TCGA submitter ids."
        )

    def _repair_row(row):
        return replace(row, cohort=cohort, rna_path=rna_tpm, clinical_path=clinical_csv)

    ds_rows = [_repair_row(x) for x in ds_rows]

    gene_sets_csv = Path(args.rna_gene_sets_csv).resolve() if args.rna_gene_sets_csv is not None else (paths.raw_rna_root / "metadata" / "hallmarks_signatures.csv").resolve()
    gene_sets = load_gene_sets_csv(gene_sets_csv)
    gene_id_to_symbol = None
    if args.gene_annotation_gtf is not None:
        gtf_path = Path(args.gene_annotation_gtf).resolve()
        if not gtf_path.exists():
            raise FileNotFoundError(f"gene_annotation_gtf not found: {gtf_path}")
        from r2wsp.data.tcga_dataset import _strip_gene_version as _sv  # type: ignore
        import gzip

        def _parse_attrs(raw: str) -> dict[str, str]:
            out: dict[str, str] = {}
            for chunk in str(raw).strip().split(";"):
                item = chunk.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                out[str(key)] = str(value).strip().strip('"')
            return out

        opener = gzip.open if gtf_path.suffix == ".gz" else open
        gene_id_to_symbol = {}
        with opener(gtf_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue
                attrs = _parse_attrs(fields[8])
                gid = attrs.get("gene_id")
                gname = attrs.get("gene_name")
                if not gid or not gname:
                    continue
                gene_id_to_symbol[_sv(gid)] = str(gname)
        if not gene_id_to_symbol:
            raise ValueError("empty gene_id_to_symbol map")

    rna_mode = "omics"
    tpm_tsv = paths.raw_rna_root / "tpm_tsv" / f"{cohort.lower()}_tpm.tsv"
    if not tpm_tsv.exists():
        raise FileNotFoundError(f"TPM TSV not found: {tpm_tsv}")
    from r2wsp.data.tcga_dataset import _CohortTPMTsv as _TPM  # type: ignore

    tpm = _TPM(tpm_tsv)
    gene_names = [
        gene_id_to_symbol.get(_strip_gene_version(g), _strip_gene_version(g)) if gene_id_to_symbol is not None else _strip_gene_version(g)
        for g in tpm.genes
    ]
    spec = build_omics_spec(gene_names, gene_sets)
    rna_omic_sizes = list(spec.omic_sizes)

    ds = TCGAMultimodalDataset(
        ds_rows,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.val_frac),
        wsi_feature_source=str(args.wsi_feature_source),
        max_tiles=int(args.max_tiles),
        rna_mode=rna_mode,
        rna_gene_sets_csv=gene_sets_csv,
        rna_gene_id_to_symbol=gene_id_to_symbol,
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_tile_budget_mode=str(args.multi_slide_tile_budget_mode),
        use_anti_features=False,
    )

    case_to_idx = {str((g[0] if isinstance(g, list) else g).case_id): i for i, g in enumerate(ds.rows)}
    train_idx = [case_to_idx[c] for c in train_ids if c in case_to_idx]
    val_idx = [case_to_idx[c] for c in val_ids if c in case_to_idx]
    test_idx = [case_to_idx[c] for c in test_ids if c in case_to_idx]
    if not train_idx or not val_idx:
        raise ValueError("empty dataset indices after asset alignment")

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=collate_tiles_and_rna,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=collate_tiles_and_rna,
    )
    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=collate_tiles_and_rna,
    )

    device = torch.device("cuda" if str(args.device) == "cuda" else "cpu")
    model = _build_model(
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        rna_mode=rna_mode,
        rna_omic_sizes=rna_omic_sizes,
        cross_modal_fusion="concat",
        use_multi_slide=bool(args.use_multi_slide),
        multi_slide_mode=str(args.multi_slide_mode),
    ).to(device)

    for n, p_ in model.named_parameters():
        if n.startswith("rna_") or ".rna_" in n:
            p_.requires_grad = False

    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    best = math.inf
    best_state = None
    best_epoch = -1
    patience = 12
    bad = 0

    def _eval(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                y = torch.tensor([pseudo[str(c)] for c in batch.case_id], dtype=torch.float32, device=device)
                rna_omics = [x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None
                if rna_omics is not None:
                    rna_omics = [torch.zeros_like(x) for x in rna_omics]
                risk, _ = model(
                    tile_tokens=batch.tile_tokens.to(device),
                    tile_xy=batch.tile_xy.to(device),
                    tile_attn_mask=batch.tile_attn_mask.to(device),
                    slide_ids=batch.slide_ids.to(device),
                    rna_vec=None,
                    rna_omics=rna_omics,
                    wsi_anti_vec=batch.wsi_anti_vec.to(device) if batch.wsi_anti_vec is not None else None,
                    rna_anti_vec=batch.rna_anti_vec.to(device) if batch.rna_anti_vec is not None else None,
                )
                losses.append(torch.mean((risk.reshape(-1) - y) ** 2).detach().cpu().item())
        return float(np.mean(losses)) if losses else float("inf")

    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in train_loader:
            y = torch.tensor([pseudo[str(c)] for c in batch.case_id], dtype=torch.float32, device=device)
            rna_omics = [x.to(device) for x in batch.rna_omics] if batch.rna_omics is not None else None
            if rna_omics is not None:
                rna_omics = [torch.zeros_like(x) for x in rna_omics]
            opt.zero_grad(set_to_none=True)
            risk, _ = model(
                tile_tokens=batch.tile_tokens.to(device),
                tile_xy=batch.tile_xy.to(device),
                tile_attn_mask=batch.tile_attn_mask.to(device),
                slide_ids=batch.slide_ids.to(device),
                rna_vec=None,
                rna_omics=rna_omics,
                wsi_anti_vec=batch.wsi_anti_vec.to(device) if batch.wsi_anti_vec is not None else None,
                rna_anti_vec=batch.rna_anti_vec.to(device) if batch.rna_anti_vec is not None else None,
            )
            loss = torch.mean((risk.reshape(-1) - y) ** 2)
            loss.backward()
            opt.step()

        val_mse = _eval(val_loader)
        train_mse = _eval(train_loader)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        (out_dir / "histories" / f"epoch_{epoch:03d}.json").write_text(json.dumps(history[-1], ensure_ascii=False) + "\n", encoding="utf-8")
        if val_mse + 1e-8 < best:
            best = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = len(history)
        best = history[-1]["val_mse"] if history else float("inf")

    best_path = out_dir / "best.pt"
    torch.save(
        {
            "model": best_state,
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best),
            "mode": "stage1_pseudorisk_wsi_only",
        },
        best_path,
    )

    test_mse = _eval(test_loader) if test_idx else float("inf")
    summary = {
        "stage": 1,
        "mode": "stage1_pseudorisk_wsi_only",
        "cohort": cohort,
        "split_dir": str(split_dir),
        "pseudo_risk_csv": str(Path(args.pseudo_risk_csv).resolve()),
        "wsi_feature_source": str(args.wsi_feature_source),
        "use_multi_slide": bool(args.use_multi_slide),
        "multi_slide_mode": str(args.multi_slide_mode),
        "multi_slide_tile_budget_mode": str(args.multi_slide_tile_budget_mode),
        "rna_mode": rna_mode,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_tiles": int(args.max_tiles),
        "n_train_cases": int(len(train_idx)),
        "n_val_cases": int(len(val_idx)),
        "n_test_cases": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best),
        "final_test_mse": float(test_mse),
        "best_path": str(best_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
