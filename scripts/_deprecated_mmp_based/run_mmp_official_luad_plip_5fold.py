from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _find_completed_dump(out_dir: Path) -> Path | None:
    matches = sorted(out_dir.glob("**/all_dumps.h5"))
    return matches[-1] if matches else None


def _append_arg(cmd: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def _resolve_data_source(args: argparse.Namespace, bridge_root: Path) -> Path:
    if args.data_source:
        return Path(args.data_source).resolve()
    return (
        bridge_root
        / "histology"
        / f"extracted_mag{int(args.patch_mag)}x_patch{int(args.patch_size)}_fp"
        / str(args.feature_name)
        / str(args.feature_dir_name)
    ).resolve()


def _resolve_results_root(args: argparse.Namespace, bridge_root: Path) -> Path:
    if args.results_root:
        return Path(args.results_root).resolve()
    if args.variant:
        base = (bridge_root / f"results_5fold_{args.variant}").resolve()
    else:
        base = (bridge_root / "results_5fold").resolve()
    if int(args.seed) != 1:
        return base.with_name(f"{base.name}_seed{int(args.seed)}")
    return base


def _build_command(
    *,
    python_executable: str,
    data_source: Path,
    omics_dir: Path,
    split_dir: Path,
    results_dir: Path,
    num_workers: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_executable,
        "training/main_survival.py",
        "--data_source",
        str(data_source),
        "--omics_dir",
        str(omics_dir),
        "--split_dir",
        str(split_dir),
        "--split_names",
        str(args.split_names),
        "--task",
        str(args.task),
        "--target_col",
        str(args.target_col),
        "--model_histo_type",
        str(args.model_histo_type),
        "--model_histo_config",
        str(args.model_histo_config),
        "--model_mm_type",
        str(args.model_mm_type),
        "--in_dim",
        str(args.in_dim),
        "--bag_size",
        str(args.bag_size),
        "--batch_size",
        str(args.batch_size),
        "--max_epochs",
        str(args.max_epochs),
        "--lr",
        str(args.lr),
        "--wd",
        str(args.wd),
        "--lr_scheduler",
        str(args.lr_scheduler),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--loss_fn",
        str(args.loss_fn),
        "--n_label_bins",
        str(args.n_label_bins),
        "--num_workers",
        str(num_workers),
        "--results_dir",
        str(results_dir),
    ]
    _append_arg(cmd, "--train_bag_size", args.train_bag_size)
    _append_arg(cmd, "--val_bag_size", args.val_bag_size)
    _append_arg(cmd, "--opt", args.opt)
    _append_arg(cmd, "--seed", args.seed)
    _append_arg(cmd, "--accum_steps", args.accum_steps)
    _append_arg(cmd, "--warmup_steps", args.warmup_steps)
    _append_arg(cmd, "--early_stopping", args.early_stopping)
    _append_arg(cmd, "--es_min_epochs", args.es_min_epochs)
    _append_arg(cmd, "--es_patience", args.es_patience)
    _append_arg(cmd, "--es_metric", args.es_metric)
    _append_arg(cmd, "--omics_modality", args.omics_modality)
    _append_arg(cmd, "--type_of_path", args.type_of_path)
    return cmd


def _run_fold(
    *,
    fold: int,
    python_executable: str,
    mmp_src: Path,
    data_source: Path,
    omics_dir: Path,
    split_dir: Path,
    results_dir: Path,
    num_workers: int,
    heartbeat_seconds: int,
    rerun: bool,
    args: argparse.Namespace,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train.log"
    done_path = _find_completed_dump(results_dir)
    if done_path is not None and not rerun:
        print(f"[fold {fold}] skip: already completed -> {done_path}")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(mmp_src)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_MAX_THREADS"] = "1"

    cmd = _build_command(
        python_executable=python_executable,
        data_source=data_source,
        omics_dir=omics_dir,
        split_dir=split_dir,
        results_dir=results_dir,
        num_workers=num_workers,
        args=args,
    )

    print(f"[fold {fold}] start")
    print(f"[fold {fold}] split_dir={split_dir}")
    print(f"[fold {fold}] results_dir={results_dir}")
    print(f"[fold {fold}] log={log_path}")
    print(f"[fold {fold}] cmd={' '.join(cmd)}")

    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"\n===== fold {fold} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_fp.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(mmp_src),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        last_output_at = time.time()
        last_heartbeat_at = time.time()
        line_count = 0

        while True:
            line = process.stdout.readline()
            if line:
                text = line.rstrip("\n")
                print(text, flush=True)
                log_fp.write(line)
                log_fp.flush()
                line_count += 1
                last_output_at = time.time()
            elif process.poll() is not None:
                break
            else:
                now = time.time()
                if now - last_heartbeat_at >= heartbeat_seconds:
                    idle = int(now - last_output_at)
                    heartbeat = f"[fold {fold}] heartbeat: running, idle_seconds={idle}, lines={line_count}, time={time.strftime('%H:%M:%S')}"
                    print(heartbeat, flush=True)
                    log_fp.write(heartbeat + "\n")
                    log_fp.flush()
                    last_heartbeat_at = now
                time.sleep(1.0)

        # Drain any remaining buffered output
        for line in process.stdout:
            print(line.rstrip("\n"), flush=True)
            log_fp.write(line)
            line_count += 1
        log_fp.flush()

    code = int(process.wait())
    done_path = _find_completed_dump(results_dir)
    if code == 0 and done_path is not None:
        print(f"[fold {fold}] done: {done_path}")
    else:
        print(f"[fold {fold}] exit_code={code}, completed_dump={done_path}")
    return code


def main() -> None:
    p = argparse.ArgumentParser(description="Run MMP LUAD 5-fold survival experiments with configurable WSI protocol and heartbeat logging.")
    p.add_argument("--mmp-root", default="/root/autodl-tmp/_refs/MMP-main")
    p.add_argument("--bridge-root", default="/root/autodl-tmp/R2wsp/data/bridges/mmp_luad_plip_luad_256_official_dss")
    p.add_argument("--cohort", default="LUAD")
    p.add_argument("--results-root", default=None)
    p.add_argument("--data-source", default=None)
    p.add_argument("--omics-dir", default=None)
    p.add_argument("--splits-root", default=None)
    p.add_argument("--feature-name", default="plip_luad_256")
    p.add_argument("--feature-dir-name", default="feats_pt")
    p.add_argument("--patch-mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--python-executable", default=sys.executable)
    p.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--heartbeat-seconds", type=int, default=60)
    p.add_argument("--rerun", action="store_true", default=False)
    p.add_argument("--split-names", default="train,test")
    p.add_argument("--task", default="LUAD_survival")
    p.add_argument("--target-col", default="dss_survival_days")
    p.add_argument("--model-histo-type", default="MIL")
    p.add_argument("--model-histo-config", default="MIL_default")
    p.add_argument("--model-mm-type", default="survpath")
    p.add_argument("--in-dim", type=int, default=512)
    p.add_argument("--bag-size", type=int, default=256)
    p.add_argument("--train-bag-size", type=int, default=None)
    p.add_argument("--val-bag-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--opt", default=None)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--loss-fn", default="nll")
    p.add_argument("--n-label-bins", type=int, default=4)
    p.add_argument("--early-stopping", type=int, default=None)
    p.add_argument("--es-min-epochs", type=int, default=None)
    p.add_argument("--es-patience", type=int, default=None)
    p.add_argument("--es-metric", default=None)
    p.add_argument("--omics-modality", default=None)
    p.add_argument("--type-of-path", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--exp-code", default=None)
    args = p.parse_args()

    mmp_root = Path(args.mmp_root).resolve()
    mmp_src = (mmp_root / "src").resolve()
    bridge_root = Path(args.bridge_root).resolve()
    results_root = _resolve_results_root(args, bridge_root)

    data_source = _resolve_data_source(args, bridge_root)
    omics_dir = Path(args.omics_dir).resolve() if args.omics_dir else (mmp_src / "data_csvs" / "rna").resolve()
    splits_root = Path(args.splits_root).resolve() if args.splits_root else (bridge_root / "splits" / "survival").resolve()
    cohort = str(args.cohort).upper()

    print("=== MMP 5-fold runner ===")
    print("mmp_src      =", mmp_src)
    print("bridge_root  =", bridge_root)
    print("cohort       =", cohort)
    print("data_source  =", data_source)
    print("omics_dir    =", omics_dir)
    print("splits_root  =", splits_root)
    print("results_root =", results_root)
    print("python_exec  =", args.python_executable)
    print("folds        =", args.folds)
    print("variant      =", args.variant)
    print("exp_code_in  =", args.exp_code)
    print("task         =", args.task)
    print("target_col   =", args.target_col)
    print("histo_type   =", args.model_histo_type)
    print("histo_config =", args.model_histo_config)
    print("mm_type      =", args.model_mm_type)
    print("in_dim       =", args.in_dim)
    print("bag_size     =", args.bag_size)
    print("train_bag    =", args.train_bag_size)
    print("val_bag      =", args.val_bag_size)
    print("batch_size   =", args.batch_size)
    print("seed         =", args.seed)
    print("max_epochs   =", args.max_epochs)
    print("lr           =", args.lr)
    print("wd           =", args.wd)
    print("scheduler    =", args.lr_scheduler)
    print("warmup_ep    =", args.warmup_epochs)
    print("early_stop   =", args.early_stopping)
    print("es_min_ep    =", args.es_min_epochs)
    print("es_patience  =", args.es_patience)
    print("num_workers  =", args.num_workers)
    print("heartbeat_s  =", args.heartbeat_seconds)
    if args.exp_code is not None or args.variant is not None:
        print("note         = upstream MMP exp_code path is broken; runner tags experiments via results_root only")
    if (
        str(args.model_mm_type).lower() == "survpath"
        and args.train_bag_size is None
        and args.val_bag_size is None
        and int(args.bag_size) > 0
    ):
        print("warning      = current args will cap both train and test bags")
        print("warning      = original MMP survpath.sh uses --bag_size -1 --train_bag_size 4096 --val_bag_size -1")

    failed: list[int] = []
    for fold in args.folds:
        split_dir = splits_root / f"TCGA_{cohort}_overall_survival_k={int(fold)}"
        fold_results_dir = results_root / f"k={int(fold)}"
        code = _run_fold(
            fold=int(fold),
            python_executable=str(args.python_executable),
            mmp_src=mmp_src,
            data_source=data_source,
            omics_dir=omics_dir,
            split_dir=split_dir,
            results_dir=fold_results_dir,
            num_workers=int(args.num_workers),
            heartbeat_seconds=int(args.heartbeat_seconds),
            rerun=bool(args.rerun),
            args=args,
        )
        if code != 0:
            failed.append(int(fold))

    print("=== Runner finished ===")
    if failed:
        print("failed_folds =", failed)
        raise SystemExit(1)
    print("failed_folds = []")


if __name__ == "__main__":
    main()
