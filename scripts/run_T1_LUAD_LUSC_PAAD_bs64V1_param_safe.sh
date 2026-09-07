#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/R2wsp
mkdir -p /root/autodl-tmp/R2wsp/outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

COHORTS="LUAD LUSC PAAD"
SEEDS="0 1 2"
FOLDS="0 1 2 3 4"

for COHORT in ${COHORTS}; do
  for s in ${SEEDS}; do
    for k in ${FOLDS}; do
      STAGE1_DIR="/root/autodl-tmp/R2wsp/outputs/twostage_pretrain_bs64_V1/${COHORT}/seed${s}/fold${k}/stage1"
      STAGE1_BEST="${STAGE1_DIR}/best.pt"
      if [ ! -f "${STAGE1_BEST}" ]; then
        python /root/autodl-tmp/R2wsp/scripts/train_censored_stage_survival.py \
          --device cuda \
          --stage 1 \
          --cohort_hint ${COHORT} \
          --wsi_feature_source uni1024 \
          --rna_mode omics \
          --rna_gene_sets_csv /root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv \
          --gene_annotation_gtf /root/autodl-tmp/gencode.v22.annotation.gtf.gz \
          --split_dir /root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol_${COHORT}/phase1_outer5/fold_${k} \
          --out_dir "${STAGE1_DIR}" \
          --seed ${s} \
          --batch_size 64 \
          --num_workers 2 \
          --epochs 50 \
          --lr 1e-4 \
          --weight_decay 1e-4 \
          --hidden_dim 256 \
          --dropout 0.15 \
          --max_tiles 256 \
          --use_multi_slide \
          --multi_slide_mode slide_attn_case_attn \
          --multi_slide_tile_budget_mode case_shared \
          || echo "[FAIL stage1] ${COHORT} s=${s} k=${k}"
      fi

      STAGE2_DIR="/root/autodl-tmp/R2wsp/outputs/twostage_param_safe_bs64_V1/${COHORT}/seed${s}/fold${k}/stage2"
      STAGE2_SUM="${STAGE2_DIR}/summary.json"
      if [ -f "${STAGE2_SUM}" ]; then
        echo "[SKIP stage2] ${STAGE2_SUM}"
        continue
      fi
      if [ ! -f "${STAGE1_BEST}" ]; then
        echo "[MISSING stage1 ckpt] ${STAGE1_BEST}"
        continue
      fi

      python /root/autodl-tmp/R2wsp/scripts/train_censored_stage_survival.py \
        --device cuda \
        --stage 2 \
        --cohort_hint ${COHORT} \
        --wsi_feature_source uni1024 \
        --rna_mode omics \
        --rna_gene_sets_csv /root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv \
        --gene_annotation_gtf /root/autodl-tmp/gencode.v22.annotation.gtf.gz \
        --split_dir /root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol_${COHORT}/phase2_independent5/fold_${k}/eval_with_censored \
        --out_dir "${STAGE2_DIR}" \
        --seed ${s} \
        --batch_size 64 \
        --num_workers 2 \
        --epochs 50 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --hidden_dim 256 \
        --dropout 0.15 \
        --max_tiles 256 \
        --use_multi_slide \
        --multi_slide_mode slide_attn_case_attn \
        --multi_slide_tile_budget_mode case_shared \
        --pretrain_checkpoint "${STAGE1_BEST}" \
        --loss_window_policy param \
        --loss_window_metric val_c_index_ema \
        --loss_window_center_mode ema \
        --loss_window_width_mode quantile \
        --loss_window_width_quantile 0.8 \
        --loss_window_transition 0.25 \
        --loss_window_width_min 0.06 \
        --loss_window_eps 0.05 \
        || echo "[FAIL stage2] ${COHORT} s=${s} k=${k}"

      sleep 2
    done
  done
done

