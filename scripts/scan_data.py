from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r2wsp.data import build_index, resolve_data_paths, scan_all_assets, summarize_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    paths = resolve_data_paths(config_path=args.config, data_root=args.data_root)
    inv = scan_all_assets(paths)
    rows = build_index(inv)
    out = {
        "paths": paths.as_dict(),
        "inventory": inv.summary(),
        "index": summarize_index(rows),
    }
    if args.json:
        sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    else:
        print("paths.data_root", out["paths"]["data_root"])
        print("inventory.svs_count", out["inventory"]["svs_count"])
        print("inventory.token_count", out["inventory"]["token_count"])
        print("inventory.rna_count", out["inventory"]["rna_count"])
        print("inventory.clinical_count", out["inventory"]["clinical_count"])
        print("index.rows", out["index"]["rows"])
        print("index.with_case_id", out["index"]["with_case_id"])
        print("index.with_svs", out["index"]["with_svs"])
        print("index.with_rna", out["index"]["with_rna"])
        print("index.with_clinical", out["index"]["with_clinical"])


if __name__ == "__main__":
    main()
