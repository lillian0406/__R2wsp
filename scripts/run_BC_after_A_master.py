#!/usr/bin/env python3
"""跑完 A 天花板 12 次后，自动顺序跑 Batch B / C1 / C2（共 45 次 5 折两窗口 × 两 eval）。
用法：直接运行本脚本（前台持有；失败时末尾会打印失败的 tag 和 --start-from 恢复）。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = "/root/autodl-tmp/venvs/tcga1126/bin/python"

# ---- Batch A 完成判据：summary.json == 12 ----
BATCH_A_DIR = PROJECT_ROOT / "outputs/censored_stage_survival_phase2/k3_grid_A_ablation/top1_win0.59-0.68_seed0"
BATCH_A_EXPECTED = 12

# ---- Batch B / C1 / C2 的三个 scheduler 副本和 tag ----
BATCHES = [
    ("BatchB (k5, win0.59-0.65 A0.99, eval_uncensored_only, 15 runs)",
     str(PROJECT_ROOT / "scripts/_sch_batchB_win0.59_0.65_A0.99_k5_evalUncens.py"),
     ["--only", "phase2", "--skip-existing"]),
    ("BatchC1 (k5, win0.59-0.68 A0.99, eval_with_censored, 15 runs)",
     str(PROJECT_ROOT / "scripts/_sch_batchC1_win0.59_0.68_A0.99_k5_evalCens.py"),
     ["--only", "phase2", "--skip-existing"]),
    ("BatchC2 (k5, win0.59-0.68 A0.99, eval_uncensored_only, 15 runs)",
     str(PROJECT_ROOT / "scripts/_sch_batchC2_win0.59_0.68_A0.99_k5_evalUncens.py"),
     ["--only", "phase2", "--skip-existing"]),
]


def wait_batch_a(timeout_s: int = 3600, poll: int = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        got = len(list(BATCH_A_DIR.glob("*/fold_*/summary.json")))
        now = time.strftime("%H:%M:%S")
        print(f"[A WAIT {now}] summary.json = {got}/{BATCH_A_EXPECTED}  elapsed {int(time.time()-t0)}s",
              flush=True)
        if got >= BATCH_A_EXPECTED:
            print(f"[A DONE {now}] Batch A 全部完成，继续后面 3 批。", flush=True)
            return
        time.sleep(poll)
    raise TimeoutError(f"Batch A 等了 {timeout_s}s 还没到 {BATCH_A_EXPECTED} 个。")


def run_sched(title: str, sch_path: str, extra: list[str]) -> int:
    cmd = [PY, sch_path, *extra]
    tag = f"[{title}]"
    print(f"\n{tag} START  cmd={' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    assert p.stdout is not None
    log_path = PROJECT_ROOT / f"outputs/_run_{Path(sch_path).stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_f:
        for raw in p.stdout:
            sys.stdout.buffer.write(raw)
            sys.stdout.flush()
            log_f.write(raw)
            log_f.flush()
    rc = p.wait()
    print(f"{tag} END  rc={rc}  log={log_path}", flush=True)
    return rc


def main() -> None:
    # 先给用户一个 Batch A 即时快照（不用等 30s 第一次 poll）
    got = len(list(BATCH_A_DIR.glob("*/fold_*/summary.json")))
    print(f"[BOOT] Batch A 当前完成度 {got}/{BATCH_A_EXPECTED}")
    if got < BATCH_A_EXPECTED:
        wait_batch_a()
    else:
        print("[BOOT] Batch A 已齐全，直接开跑 Batch B/C1/C2")

    for title, sch, extra in BATCHES:
        rc = run_sched(title, sch, extra)
        if rc != 0:
            raise SystemExit(f"[FAIL] {title} rc={rc}  —— 恢复：把上面 scheduler 失败末尾的 --start-from 粘到原命令。")

    print("\n[ALL BATCHES DONE]")


if __name__ == "__main__":
    main()
