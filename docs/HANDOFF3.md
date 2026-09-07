# 项目交接说明（R2wsp｜换服务器与续跑向）

更新时间：2026-09-03（Asia/Shanghai）

## 0. 本交接文档用途

- 用于在更换服务器后，确保新环境能“最小成本、零口径歧义”地恢复项目运行与论文口径一致性。
- 覆盖：迁移资产清单、数据/泄露审计结论、关键口径锁定、当前实验状态、常见坑与续跑步骤。

## 1. 项目一句话概述

- 任务：多癌种 TCGA 的 WSI+RNA 生存预测（高删失），采用 Cox 排序学习（Cox-NPLL）作为核心目标。
- 核心方法（论文最终口径）：two-stage + pullback + validation-driven window scheduler（param-safe），目标是提升 C-index 并降低跨 seed/fold 方差。
- 评估协议：严格 5 folds × 3 seeds（N=15），以 stage2 的 `best_test_c_index` 作为汇总指标。

## 2. 最小可复现资产（迁移清单）

### 2.1 必须迁移（缺一不可）

- `outputs/`（模型权重、history、summary、日志）
- `data/splits/`（所有 split 协议与 CSV）
- `data/wsi_features/`（UNI 特征；本项目离线特征训练模式，不再依赖 raw svs）
- `data/raw_rna/`（RNA 表达 + hallmark + gtf 等元数据）

### 2.2 严禁迁移/无需保留（已清理）

- `data/downloads/`
- `data/raw_svs/`（原始 svs 已删除；但训练脚本路径校验需要目录存在，见 6.1）
- `data/gdc.zip`

## 3. 迁移后审计结论（可释放旧服务器的依据）

### 3.1 Split 泄露检查（通过）

- 对 6 个主 cohort（BRCA/LUAD/LUSC/PAAD/UCEC/BLCA）的 `phase2_independent5`：
  - train/test `case_id` 无交集
  - train/test `slide_id` 无交集
- 结论：无数据泄露风险（fit-on-train-only 口径可成立）。

### 3.2 UNI 特征缺漏（历史遗留，不是迁移漏拷贝）

- BRCA split 内存在 `slide_id=TCGA-A8-A07F-01Z-00-DX1`，在 UNI 特征库中找不到 `.h5`。
- 已在旧服务器同目录执行 `find ... -name 'TCGA-A8-A07F-01Z-00-DX1*'` 为空。
- 结论：该缺失为历史遗留，不是迁移缺漏。

## 4. 论文/数据口径锁定（新助手必须按此执行）

### 4.1 UCEC split 版本

- 当前默认/主路径：`data/splits/censored_stage_protocol_UCEC/phase2_independent5`，unique `case_id=387`。
- 变体：
  - `censored_stage_protocol_UCEC_470`：390
  - `censored_stage_protocol_UCEC_470_r2`：387（回到 387）
- 论文主表默认使用 387（因工程脚本大量默认指向 `censored_stage_protocol_UCEC`）。

### 4.2 表 5.1 的 “SVS 切片数”统计口径

- 写 modeling（训练实际使用）：从 `phase2_independent5/fold_0/eval_with_censored/train.csv + test.csv` 的 `slide_id` 去重统计得到。
- 不用 inventory/audit 总表口径。

### 4.3 BRCA_651 命名来源（防误解）

- `BRCA_651` 是早期误算残留（曾按 300+315≈651 估算）。
- 实际可用为 577 case / 82 events（大量 case 因 bug/缺模态不可用被剔除）。
- 后续以 split/summary.json 的实际统计为准，不以目录名数字为准。

## 5. 当前关键实验状态（你关心的 bs64 对齐）

### 5.1 动机

- 怀疑 batch=64 性能差与学习率调度/正则设置不匹配有关（以及固定 epoch 下 step 不足导致欠拟合）。
- 目标：对齐参考 recipe：`lr=1e-4 + cosine`、`AdamW wd=1e-5`、`epochs=20`、`batch=64`、Cox loss。

### 5.2 已完成产物（存在于 outputs）

- BRCA Stage1（已完成 15/15）：
  - `outputs/BRCA_pretrain_bs64_ep20_cosine_wd1e5/BRCA/seed{0,1,2}/fold{0..4}/stage1/summary.json`
- BRCA Stage2 param_safe（当前 3/15，仅 seed0 fold0-2）：
  - `outputs/BRCA_param_safe_bs64_ep20_cosine_wd1e5/BRCA/seed0/fold{0,1,2}/stage2/summary.json`

### 5.3 CPU 机的结论

- 当前便宜 CPU 实例无 GPU（`torch.cuda.is_available=False`），并且 CPU 跑 Stage2 出现频繁 segfault，无法可靠补齐 15/15。
- 结论：需要换到带 GPU 的实例续跑。

## 6. 新服务器续跑前的两处必踩坑（必须先修）

### 6.1 data/raw_svs 路径校验

- 训练脚本会做路径校验，即使不需要 raw svs，也会因目录不存在而失败。
- 处理：创建空目录即可：`mkdir -p data/raw_svs`。

### 6.2 gencode gtf.gz 损坏（已发现过）

- `data/raw_rna/metadata/gencode.v22.annotation.gtf.gz` 曾出现 gzip EOF（训练会报 `EOFError`）。
- 新服务器迁移后必须先验证：`gzip -t data/raw_rna/metadata/gencode.v22.annotation.gtf.gz`。
- 若失败，需要用正确副本覆盖再跑（以 gzip -t 通过为准）。

## 7. 新服务器选择建议（性价比）

- 首选：3080 Ti 12GB 的低价机器（优先低占用 + 可扩容大）。
- 若必须在 CUDA 12.x 兼容性与价格间折中：优先 CUDA 12.2/12.4 的机型。
- 新机验收（必须做）：
  - `nvidia-smi`
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.__version__)"`（应为 True 且 device_count>=1）

## 8. 续跑任务：补齐 BRCA bs64 param_safe stage2 到 15/15

### 8.1 需要跑哪些组合

- seed=0..2，fold=0..4，共 15 个 stage2。
- 若 `stage2/summary.json` 已存在则跳过（目前已存在 3 个）。

### 8.2 关键参数（与论文/交接保持一致）

- `--batch_size 64`
- `--epochs 20`
- `--lr 1e-4 --lr_scheduler cosine --lr_min 0.0`
- `--weight_decay 1e-5`
- window=param_safe 等价参数（wrapper 透传口径）：
  - `--loss_window_policy param`
  - `--loss_window_metric val_c_index_ema`
  - `--loss_window_center_mode ema`
  - `--loss_window_width_mode quantile --loss_window_width_quantile 0.8`
  - `--loss_window_transition 0.25`
  - `--loss_window_width_min 0.06`
  - `--loss_window_eps 0.05`

## 9. 论文写作风险点（必须强调给新助手）

- no-window baseline 的口径风险：若设置 w≡1，则 (1-w)=0，pullback 会被关闭；论文里必须区分“关窗但保留 pullback”与“完全不门控且不回拉”。

