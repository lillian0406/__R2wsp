#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SPLIT_ROOT="${ROOT_DIR}/data/splits/_paper5_4_censor_bias/LUAD/fold0_seed0"
STAGE1_CKPT="${ROOT_DIR}/outputs/twostage_pretrain_bs16_V1/LUAD/seed0/fold0/stage1/best.pt"
RNA_CSV="${ROOT_DIR}/data/raw_rna/metadata/hallmarks_signatures.csv"
GTF="${ROOT_DIR}/data/raw_rna/metadata/gencode.v22.annotation.gtf.gz"

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TILES="${MAX_TILES:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"

run_one () {
  local method="$1"
  local variant="$2"
  local out_dir="$3"
  local split_dir="${SPLIT_ROOT}/${variant}"

  local sum="${out_dir}/LUAD/seed0/fold0/stage2/summary.json"
  if [ -f "${sum}" ]; then
    echo "[SKIP] ${method} ${variant} ${sum}"
    return 0
  fi

  echo "[RUN] ${method} ${variant} out=${out_dir}"

  if [ "${method}" = "fixed_off" ]; then
    python -u scripts/train_censored_stage_survival.py \
      --device "${DEVICE}" \
      --stage 2 \
      --out_dir "${out_dir}" \
      --wsi_feature_source uni1024 \
      --split_dir "${split_dir}" \
      --target_col dss_survival_days \
      --rna_mode omics \
      --rna_gene_sets_csv "${RNA_CSV}" \
      --gene_annotation_gtf "${GTF}" \
      --pool_method attention \
      --gate_enabled auto \
      --use_multi_slide \
      --multi_slide_mode slide_attn_case_attn \
      --multi_slide_tile_budget_mode case_shared \
      --max_tiles "${MAX_TILES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 0 \
      --val_split_mode survival_stratified \
      --val_time_bins 4 \
      --selection_metric val_c_index_ema \
      --val_ema_decay 0.6 \
      --selection_min_epochs 1 \
      --early_stop_patience 1000 \
      --hidden_dim 256 \
      --dropout 0.15 \
      --pretrain_checkpoint "${STAGE1_CKPT}" \
      --loss_window_metric val_c_index_ema \
      --loss_window_policy param \
      --loss_window_lower 0.0 \
      --loss_window_upper 1.0 \
      --loss_window_k 50.0 \
      --loss_window_A 1.0 \
      --loss_window_eps 1.0 \
      --loss_window_center_mode fixed \
      --loss_window_width_mode fixed \
      --loss_window_transition 0.0 \
      --loss_window_K 0.25
  else
    python -u scripts/train_censored_stage_survival.py \
      --device "${DEVICE}" \
      --stage 2 \
      --out_dir "${out_dir}" \
      --wsi_feature_source uni1024 \
      --split_dir "${split_dir}" \
      --target_col dss_survival_days \
      --rna_mode omics \
      --rna_gene_sets_csv "${RNA_CSV}" \
      --gene_annotation_gtf "${GTF}" \
      --pool_method attention \
      --gate_enabled auto \
      --use_multi_slide \
      --multi_slide_mode slide_attn_case_attn \
      --multi_slide_tile_budget_mode case_shared \
      --max_tiles "${MAX_TILES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --seed 0 \
      --val_split_mode survival_stratified \
      --val_time_bins 4 \
      --selection_metric val_c_index_ema \
      --val_ema_decay 0.6 \
      --selection_min_epochs 1 \
      --early_stop_patience 1000 \
      --hidden_dim 256 \
      --dropout 0.15 \
      --pretrain_checkpoint "${STAGE1_CKPT}" \
      --loss_window_metric val_c_index_ema \
      --loss_window_policy param \
      --loss_window_lower 0.58 \
      --loss_window_upper 0.65 \
      --loss_window_k 50.0 \
      --loss_window_A 0.99 \
      --loss_window_eps 0.05 \
      --loss_window_center_mode ema \
      --loss_window_width_mode quantile \
      --loss_window_width_quantile 0.8 \
      --loss_window_width_min 0.06 \
      --loss_window_width_max 0.3 \
      --loss_window_transition 0.25 \
      --loss_window_K 0.25
  fi

  local run_dir="${out_dir}/LUAD/seed0/fold0/stage2"
  local final_ckpt="${run_dir}/final.pt"
  if [ ! -f "${final_ckpt}" ]; then
    echo "[ERROR] missing final checkpoint: ${final_ckpt}"
    return 1
  fi
  python -u scripts/export_censored_case_risk.py \
    --run_dir "${run_dir}" \
    --ckpt_path "${final_ckpt}" \
    --export_tag "paper_5_4_final" \
    --splits test \
    --device cpu
}

OUT_BASE="${ROOT_DIR}/outputs/paper_5_4_censor_bias/LUAD/fold0_seed0"
mkdir -p "${OUT_BASE}"

for method in fixed_off param_safe; do
  for variant in u_only all_correct all_masked; do
    out_dir="${OUT_BASE}/${method}_${variant}"
    run_one "${method}" "${variant}" "${out_dir}"
  done
done
