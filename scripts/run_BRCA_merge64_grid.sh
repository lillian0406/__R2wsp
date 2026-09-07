#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/R2wsp"
cd "${ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

COHORT="BRCA"
SEEDS="0 1 2"
FOLDS="0 1 2 3 4"
BATCHES="8 16 32 64"
WINS="param_safe param_best_mix fixed_off onek twok"

EPOCHS="${EPOCHS:-50}"
WD="${WD:-1e-4}"
MAX_TILES="${MAX_TILES:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"

TAG="${TAG:-merge64_v1}"

for BS in ${BATCHES}; do
  EXP1="twostage_pretrain_bs${BS}_ep${EPOCHS}_${TAG}"
  echo "[STAGE1] batch=${BS} exp=${EXP1}"
  for s in ${SEEDS}; do
    for k in ${FOLDS}; do
      CKPT="outputs/${EXP1}/${COHORT}/seed${s}/fold${k}/stage1/best.pt"
      if [ -f "${CKPT}" ]; then
        echo "[SKIP stage1] ${CKPT}"
        continue
      fi
      python scripts/train_sota_stage_survival.py \
        --exp_name "${EXP1}" \
        --cohort "${COHORT}" --stage 1 --seed "${s}" --fold "${k}" \
        --epochs "${EPOCHS}" --batch_size "${BS}" --num_workers "${NUM_WORKERS}" \
        --weight_decay "${WD}" --max_tiles "${MAX_TILES}" \
        || echo "[FAIL stage1] bs=${BS} s=${s} k=${k}"
      sleep 2
    done
  done

  echo "[STAGE2] batch=${BS} reuse_pretrain=${EXP1}"
  for WIN in ${WINS}; do
    EXP2="twostage_${WIN}_bs${BS}_ep${EPOCHS}_${TAG}"
    for s in ${SEEDS}; do
      for k in ${FOLDS}; do
        SUM="outputs/${EXP2}/${COHORT}/seed${s}/fold${k}/stage2/summary.json"
        if [ -f "${SUM}" ]; then
          echo "[SKIP stage2] ${SUM}"
          continue
        fi
        python scripts/train_sota_stage_survival.py \
          --exp_name "${EXP2}" \
          --pretrain_exp_name "${EXP1}" \
          --cohort "${COHORT}" --stage 2 --seed "${s}" --fold "${k}" --window "${WIN}" \
          --epochs "${EPOCHS}" --batch_size "${BS}" --num_workers "${NUM_WORKERS}" \
          --weight_decay "${WD}" --max_tiles "${MAX_TILES}" \
          || echo "[FAIL stage2] bs=${BS} win=${WIN} s=${s} k=${k}"
        sleep 2
      done
    done
  done
done

