#!/usr/bin/env bash
set -euo pipefail

COHORT="${1:-}"
SEEDS="${2:-0}"
FOLDS="${3:-0 1 2 3 4}"

if [ -z "${COHORT}" ]; then
  echo "usage: $0 <COHORT> [\"SEEDS\"] [\"FOLDS\"]" >&2
  exit 2
fi

cd /root/autodl-tmp/R2wsp
mkdir -p outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

for s in ${SEEDS}; do
  for k in ${FOLDS}; do
    bash /root/autodl-tmp/R2wsp/scripts/run_onefold_pseudorisk_stage1_wsi_only_then_stage2.sh "${COHORT}" "${k}" "${s}"
  done
done

