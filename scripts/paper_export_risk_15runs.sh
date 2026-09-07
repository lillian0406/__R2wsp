#!/usr/bin/env bash
set -euo pipefail

COHORT="${1:-LUAD}"
COHORT="$(echo "$COHORT" | tr '[:lower:]' '[:upper:]')"

STAGE1_ROOT="${STAGE1_ROOT:-outputs/twostage_pretrain_bs16_V1}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs/twostage_param_safe_bs16_V1}"

DEVICE="${DEVICE:-cpu}"
EXPORT_BS="${EXPORT_BS:-8}"

for s in 0 1 2; do
  for f in 0 1 2 3 4; do
    P1_DIR="$STAGE1_ROOT/$COHORT/seed${s}/fold${f}/stage1"
    P2_DIR="$STAGE2_ROOT/$COHORT/seed${s}/fold${f}/stage2"

    PHASE2_SPLIT="/root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol_${COHORT}/phase2_independent5/fold_${f}/eval_with_censored"
    P1_CSV="$P1_DIR/case_risk_exports/best_phase2test/test_cases.csv"
    if [ ! -f "$P1_CSV" ]; then
      python scripts/export_censored_case_risk.py \
        --run_dir "$P1_DIR" \
        --ckpt_path "$P1_DIR/best.pt" \
        --export_tag best_phase2test \
        --split_dir_override "$PHASE2_SPLIT" \
        --target_col_override dss_survival_days \
        --splits test \
        --export_batch_size "$EXPORT_BS" \
        --device "$DEVICE"
    fi

    P2B_CSV="$P2_DIR/case_risk_exports/best/test_cases.csv"
    if [ ! -f "$P2B_CSV" ]; then
      python scripts/export_censored_case_risk.py \
        --run_dir "$P2_DIR" \
        --ckpt_path "$P2_DIR/best.pt" \
        --splits test \
        --export_batch_size "$EXPORT_BS" \
        --device "$DEVICE"
    fi

    P2F_CSV="$P2_DIR/case_risk_exports/final/test_cases.csv"
    if [ ! -f "$P2F_CSV" ]; then
      python scripts/export_censored_case_risk.py \
        --run_dir "$P2_DIR" \
        --ckpt_path "$P2_DIR/final.pt" \
        --splits test \
        --export_batch_size "$EXPORT_BS" \
        --device "$DEVICE"
    fi
  done
done
