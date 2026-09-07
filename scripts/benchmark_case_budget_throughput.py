from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def build_command(args: argparse.Namespace, budget_mode: str, num_workers: int, out_dir: Path) -> list[str]:
    cmd = [
        args.python_bin,
        "scripts/train_direct_wsi_rna_survival.py",
        "--split_dir",
        args.split_dir,
        "--wsi_feature_source",
        args.wsi_feature_source,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(num_workers),
        "--out_dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--model_variant",
        "baseline",
        "--use_multi_slide",
        "--multi_slide_mode",
        "slide_attn_case_attn",
        "--multi_slide_tile_budget_mode",
        budget_mode,
        "--max_tiles",
        str(args.max_tiles),
        "--wsi_geo_type",
        "none",
        "--rna_mode",
        "vec",
        "--rna_geo_type",
        "none",
    ]
    if args.pin_memory:
        cmd.append("--pin_memory")
    return cmd


def run_once(args: argparse.Namespace, budget_mode: str, num_workers: int) -> dict[str, object]:
    run_name = f"bench_{budget_mode}_nw{num_workers}_seed{args.seed}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, budget_mode, num_workers, out_dir)
    env = dict(**subprocess.os.environ, CUDA_VISIBLE_DEVICES=args.cuda_visible_devices)

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=args.repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_time_sec = time.perf_counter() - start

    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    row = {
        "run_name": run_name,
        "budget_mode": budget_mode,
        "num_workers": num_workers,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_tiles": args.max_tiles,
        "wall_time_sec": round(wall_time_sec, 3),
        "exit_code": proc.returncode,
        "final_test_c_index": summary.get("final_test_c_index"),
        "summary_path": str(summary_path),
        "stdout_path": str(out_dir / "benchmark_stdout.txt"),
        "stderr_path": str(out_dir / "benchmark_stderr.txt"),
    }
    (out_dir / "benchmark_stdout.txt").write_text(proc.stdout)
    (out_dir / "benchmark_stderr.txt").write_text(proc.stderr)
    (out_dir / "benchmark_meta.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark case_shared vs per_slide throughput.")
    parser.add_argument("--repo_root", default="/root/autodl-tmp/R2wsp")
    parser.add_argument("--python_bin", default="/root/autodl-tmp/venvs/tcga1126/bin/python")
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--wsi_feature_source", required=True)
    parser.add_argument("--output_root", default="/root/autodl-tmp/R2wsp/outputs/throughput_benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tiles", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    args = parser.parse_args()

    rows = []
    for budget_mode in ("case_shared", "per_slide"):
        for num_workers in (0, 4):
            rows.append(run_once(args, budget_mode, num_workers))

    output_root = Path(args.output_root)
    csv_path = output_root / "throughput_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "throughput_report.md"
    lines = [
        "# Throughput Benchmark",
        "",
        "| run_name | budget_mode | num_workers | wall_time_sec | exit_code | final_test_c_index |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['budget_mode']} | {row['num_workers']} | {row['wall_time_sec']} | {row['exit_code']} | {row['final_test_c_index']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
