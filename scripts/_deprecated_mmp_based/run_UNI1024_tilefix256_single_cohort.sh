#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_UNI1024_tilefix256_single_cohort.sh
# -----------------------------------------------------------------------------
# 手动挂单癌种 UNI1024 tilefix256 spatial 特征提取（非全自动）。
#
# 用法：
#   bash scripts/run_UNI1024_tilefix256_single_cohort.sh BRCA
#   bash scripts/run_UNI1024_tilefix256_single_cohort.sh KIRC
#   ...
#
# 硬约束（用户明确要求）：
#   * 一次只跑一个 cohort，禁止 for 循环批量全自动
#   * 参数严格对齐 LUAD 已跑通的 pipeline：mag 20x / tile 256 / stride 256
#     / spatial / target_mpp 0.5
#   * GPU 显存保守使用：batch_size=32（3090 上 ≈ 12GB 峰值，前台留 ≥ 50% 空间）
#   * 产物独立隔离目录，不覆盖 PLIP / DINO
#
# 跑完后还需要手动跑一次 bridge：
#   bash scripts/run_UNI1024_bridge_single_cohort.sh BRCA
# （生成官方 split 的 filtered CSV 和 per-fold case/slide 计数，供训练使用）
# =============================================================================

CANCER="${1:-}"
if [[ -z "$CANCER" ]]; then
  echo "Usage: $0 <CANCER>  # e.g. BRCA / BLCA / COADREAD / KIRC / LUAD / STAD" >&2
  exit 2
fi
CANCER="${CANCER^^}"

# ---------------------------------------------------------------- paths ------
REPO_ROOT="/root/autodl-tmp/R2wsp"
PY="/root/autodl-tmp/venvs/tcga1126/bin/python"
MMP_SPLIT_ROOT="/root/autodl-tmp/_refs/MMP-main"                     # MMP official splits
UNI_CKPT="${REPO_ROOT}/data/UNI/pytorch_model.bin"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/extract_uni1024_tile256_${CANCER}_${TS}.log"

# ----------------------------------------------------- extraction params ----
PATCH_MAG=20
PATCH_SIZE=256
TARGET_MPP=0.5
STRIDE_PX=256
MIN_TISSUE=0.2
MAX_TILES=0                # keep all tissue tiles that pass tissue fraction
PROGRESS_EVERY=100
N_FOLDS=5
FEATURE_NAME="uni1024"
DEVICE="cuda:0"
BATCH_SIZE=32              # conservative; safe on 3090 ~ 12GB peak

# ---------------------------------------------------------------- safety ----
cd "${REPO_ROOT}"
echo "[run_UNI_single_cohort] cohort=${CANCER}" | tee -a "${LOG_FILE}"
echo "[run_UNI_single_cohort] log  -> ${LOG_FILE}"
echo "[run_UNI_single_cohort] PY   -> ${PY}"
echo "[run_UNI_single_cohort] CKPT -> ${UNI_CKPT}"
echo "[run_UNI_single_cohort] batch=${BATCH_SIZE}  patch_mag=${PATCH_MAG}  patch_size=${PATCH_SIZE}  stride=${STRIDE_PX}"
echo "[run_UNI_single_cohort] 下一个要手动挂的 cohort（跑完这一个再回来改参数）：按本地 raw_svs 覆盖度由高到低 = BRCA -> KIRC -> COADREAD -> LUAD_missing_10 -> (BLCA/STAD 缺 raw_svs)"

nohup "${PY}" scripts/extract_mmp_official_survival_uni_features.py \
  --config "${REPO_ROOT}/configs/data_paths.yaml" \
  --mmp-root "${MMP_SPLIT_ROOT}" \
  --cohort "${CANCER}" \
  --n-folds "${N_FOLDS}" \
  --uni-checkpoint "${UNI_CKPT}" \
  --feature-name "${FEATURE_NAME}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --patch-mag "${PATCH_MAG}" \
  --patch-size "${PATCH_SIZE}" \
  --target-mpp "${TARGET_MPP}" \
  --stride-px "${STRIDE_PX}" \
  --min-tissue-fraction "${MIN_TISSUE}" \
  --max-tiles "${MAX_TILES}" \
  --progress-every-batches "${PROGRESS_EVERY}" \
  --overwrite 2>&1 | tee -a "${LOG_FILE}"
RC="${PIPESTATUS[0]}"

echo "[run_UNI_single_cohort] done cohort=${CANCER}  rc=${RC}" | tee -a "${LOG_FILE}"
exit "${RC}"
