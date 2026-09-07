#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_UNI1024_bridge_single_cohort.sh
# -----------------------------------------------------------------------------
# 手动挂单癌种 UNI1024 -> MMP official split bridge（过滤 feature 已存在的
# slide，生成 train/test/val CSV + per-fold case/slide/censor 计数）。
#
# 用法：先跑 extract（上面那个脚本），再：
#   bash scripts/run_UNI1024_bridge_single_cohort.sh BRCA
#
# 产物：data/bridges/mmp_<cohort>_uni1024_official_dss/
#   ├─ splits/survival/TCGA_<COHORT>_overall_survival_k=0..4/{train,test,val}.csv
#   └─ split_summary.json  （含 n_train_cases, n_test_slides, censorship_counts 等）
# =============================================================================

CANCER="${1:-}"
if [[ -z "$CANCER" ]]; then
  echo "Usage: $0 <CANCER>  # e.g. BRCA / BLCA / COADREAD / KIRC / LUAD / STAD" >&2
  exit 2
fi
CANCER="${CANCER^^}"

REPO_ROOT="/root/autodl-tmp/R2wsp"
PY="/root/autodl-tmp/venvs/tcga1126/bin/python"
MMP_SPLIT_ROOT="/root/autodl-tmp/_refs/MMP-main"

PATCH_MAG=20
PATCH_SIZE=256
FEATURE_NAME="uni1024"
FEATURE_ROOT="${REPO_ROOT}/data/wsi_features/extracted_mag${PATCH_MAG}x_patch${PATCH_SIZE}_fp/${FEATURE_NAME}/feats_h5"
OUT_ROOT="${REPO_ROOT}/data/bridges/mmp_${CANCER,,}_${FEATURE_NAME}_official_dss"
N_FOLDS=5
TARGET_COL="survival_days"

cd "${REPO_ROOT}"
if [[ ! -d "${FEATURE_ROOT}" ]]; then
  echo "FEATURE_ROOT not found: ${FEATURE_ROOT}" >&2
  echo "  请先运行  bash scripts/run_UNI1024_tilefix256_single_cohort.sh ${CANCER}" >&2
  exit 3
fi

"${PY}" scripts/prepare_mmp_official_survival_features.py \
  --mmp-root "${MMP_SPLIT_ROOT}" \
  --feature-root "${FEATURE_ROOT}" \
  --feature-suffix ".h5" \
  --cohort "${CANCER}" \
  --n-folds "${N_FOLDS}" \
  --target-col "${TARGET_COL}" \
  --out-root "${OUT_ROOT}"

RC=$?
echo "[bridge done] cohort=${CANCER}  out=${OUT_ROOT}  rc=${RC}"
exit "${RC}"
