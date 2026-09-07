from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GeneSetCollection:
    names: list[str]
    gene_sets: list[list[str]]

    def __len__(self) -> int:
        return len(self.names)


def load_gene_sets_csv(path: str | Path) -> GeneSetCollection:
    p = Path(path)
    df = pd.read_csv(p)
    names: list[str] = []
    sets: list[list[str]] = []
    for col in df.columns:
        raw = df[col].dropna().astype(str).tolist()
        genes = [g.strip() for g in raw if str(g).strip()]
        if not genes:
            continue
        names.append(str(col))
        sets.append(genes)
    if not names:
        raise ValueError(f"no gene sets found in csv: {p}")
    return GeneSetCollection(names=names, gene_sets=sets)

