#!/usr/bin/env bash
set -euo pipefail

COHORT="${1:-}"
FOLD="${2:-}"
SEED="${3:-0}"

if [ -z "${COHORT}" ] || [ -z "${FOLD}" ]; then
  echo "usage: $0 <COHORT> <FOLD> [SEED]" >&2
  exit 2
fi

cd /root/autodl-tmp/R2wsp
mkdir -p outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

COHORT_UPPER="$(echo "${COHORT}" | tr '[:lower:]' '[:upper:]')"
SPLIT_DIR="/root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol_${COHORT_UPPER}/phase2_independent5/fold_${FOLD}/eval_with_censored"

TEACHER_DIR="/root/autodl-tmp/R2wsp/outputs/pseudorisk_teacher_rna_cox/${COHORT_UPPER}/seed${SEED}/fold${FOLD}"
PRETRAIN_DIR="/root/autodl-tmp/R2wsp/outputs/pseudorisk_stage1_wsi_only/${COHORT_UPPER}/seed${SEED}/fold${FOLD}/stage1"
STAGE2_DIR="/root/autodl-tmp/R2wsp/outputs/pseudorisk_stage2_param_safe/${COHORT_UPPER}/seed${SEED}/fold${FOLD}/stage2"

PSEUDO_CSV="${TEACHER_DIR}/pseudo_risk.csv"

if [ ! -f "${PSEUDO_CSV}" ]; then
  python /root/autodl-tmp/R2wsp/scripts/build_pseudorisk_rna_cox.py \
    --split_dir "${SPLIT_DIR}" \
    --cohort "${COHORT_UPPER}" \
    --target_col dss_survival_days \
    --rna_gene_sets_csv /root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv \
    --gene_annotation_gtf /root/autodl-tmp/gencode.v22.annotation.gtf.gz \
    --out_csv "${PSEUDO_CSV}" \
    --seed "${SEED}"
fi

if [ ! -f "${PRETRAIN_DIR}/best.pt" ]; then
  python /root/autodl-tmp/R2wsp/scripts/train_stage1_pseudorisk_wsi_only.py \
    --split_dir "${SPLIT_DIR}" \
    --cohort "${COHORT_UPPER}" \
    --pseudo_risk_csv "${PSEUDO_CSV}" \
    --out_dir "${PRETRAIN_DIR}" \
    --seed "${SEED}" \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 2 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --dropout 0.15 \
    --max_tiles 256 \
    --use_multi_slide \
    --multi_slide_mode slide_attn_case_attn \
    --multi_slide_tile_budget_mode case_shared \
    --wsi_feature_source uni1024 \
    --rna_gene_sets_csv /root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv \
    --gene_annotation_gtf /root/autodl-tmp/gencode.v22.annotation.gtf.gz
fi

if [ ! -f "${STAGE2_DIR}/summary.json" ]; then
  python /root/autodl-tmp/R2wsp/scripts/train_censored_stage_survival.py \
    --device cuda \
    --stage 2 \
    --cohort_hint "${COHORT_UPPER}" \
    --wsi_feature_source uni1024 \
    --rna_mode omics \
    --rna_gene_sets_csv /root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv \
    --gene_annotation_gtf /root/autodl-tmp/gencode.v22.annotation.gtf.gz \
    --split_dir "${SPLIT_DIR}" \
    --out_dir "${STAGE2_DIR}" \
    --seed "${SEED}" \
    --batch_size 16 \
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
    --pretrain_checkpoint "${PRETRAIN_DIR}/best.pt" \
    --loss_window_policy param \
    --loss_window_metric val_c_index_ema \
    --loss_window_center_mode ema \
    --loss_window_width_mode quantile \
    --loss_window_width_quantile 0.8 \
    --loss_window_transition 0.25 \
    --loss_window_width_min 0.06 \
    --loss_window_eps 0.05
fi

