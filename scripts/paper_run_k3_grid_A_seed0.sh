#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/R2wsp
mkdir -p outputs/_runner_logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

OUT_ROOT="${OUT_ROOT:-outputs/k3_grid_A_param_safe_bs16_ep50_seed0}"
RNA_CSV="${RNA_CSV:-/root/autodl-tmp/R2wsp/data/raw_rna/metadata/hallmarks_signatures.csv}"
GTF="${GTF:-/root/autodl-tmp/gencode.v22.annotation.gtf.gz}"
NUM_WORKERS="${NUM_WORKERS:-0}"

seed=0

cohorts=(BRCA LUAD LUSC PAAD UCEC BLCA)
As=(0.990 0.995 1.000 1.005)

for cohort in "${cohorts[@]}"; do
  for A in "${As[@]}"; do
    for fold in 0 1 2 3 4; do
      run_out="${OUT_ROOT}/A${A}/${cohort}/seed${seed}/fold${fold}/stage2"
      sum_json="${run_out}/summary.json"
      if [ -f "${sum_json}" ]; then
        echo "[k3_grid_A] [SKIP] cohort=${cohort} A=${A} seed=${seed} fold=${fold}"
        continue
      fi

      split_dir="/root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol_${cohort}/phase2_independent5/fold_${fold}/eval_with_censored"

      stage1_root=""
      if [ "${cohort}" = "BRCA" ]; then
        stage1_root="outputs/twostage_pretrain_bs16_ep50_merge64_v1"
      elif [ "${cohort}" = "UCEC" ]; then
        stage1_root="outputs/UCEC_pretrain_bs16_ep50"
      elif [ "${cohort}" = "BLCA" ]; then
        stage1_root="outputs/BLCA_pretrain_bs16_ep50"
      else
        stage1_root="outputs/twostage_pretrain_bs16_V1"
      fi
      ckpt="${stage1_root}/${cohort}/seed${seed}/fold${fold}/stage1/best.pt"

      mkdir -p "${run_out}"
      echo "[k3_grid_A] [RUN] cohort=${cohort} A=${A} seed=${seed} fold=${fold} out=${run_out}"

      python -u scripts/train_censored_stage_survival.py \
        --device cuda \
        --stage 2 \
        --out_dir "${run_out}" \
        --cohort_hint "${cohort}" \
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
        --max_tiles 256 \
        --batch_size 16 \
        --num_workers "${NUM_WORKERS}" \
        --epochs 50 \
        --lr 1e-4 \
        --lr_scheduler none \
        --lr_min 0.0 \
        --weight_decay 1e-4 \
        --seed ${seed} \
        --val_split_mode survival_stratified \
        --val_time_bins 4 \
        --selection_metric val_c_index_ema \
        --val_ema_decay 0.6 \
        --selection_min_epochs 8 \
        --early_stop_patience 12 \
        --hidden_dim 256 \
        --dropout 0.15 \
        --pretrain_checkpoint "${ckpt}" \
        --loss_window_metric val_c_index_ema \
        --loss_window_policy param \
        --loss_window_lower 0.58 \
        --loss_window_upper 0.65 \
        --loss_window_k 50.0 \
        --loss_window_A "${A}" \
        --loss_window_eps 0.05 \
        --loss_window_center_mode ema \
        --loss_window_width_mode quantile \
        --loss_window_width_quantile 0.8 \
        --loss_window_width_min 0.06 \
        --loss_window_width_max 0.3 \
        --loss_window_transition 0.25 \
        --loss_window_K 0.25
    done
  done
done
