# 项目交接说明（R2wsp）

更新时间：2026-09-02（Asia/Shanghai）

## 1. 项目目标与口径

- 任务：多癌种 two-stage censored survival（Stage1 预训练 + Stage2 含删失）评测与论文数字校准
- 关键口径
  - 论文表“病例数（Case）”：以 `phase2_independent5` split（train/val/test union 的 unique cases）为准
  - 指标：以各 run `outputs/**/stage2/summary.json` 中的 `best_test_c_index` 为准
  - std：与现有汇总保持一致，使用 ddof=0（population std）

## 2. 关键工程变更（已落盘）

### 2.1 Stage2 支持 cosine scheduler（对齐参考论文训练设定）

目的：支持 `lr=1e-4 + cosine decay`，并在 history/summary 中记录 `lr` 与调度器信息，便于审计复现。

- 修改文件：
  - `scripts/train_censored_stage_survival.py`
    - 新增参数：`--lr_scheduler {none,cosine}`、`--lr_min`
    - 训练期：当 `lr_scheduler=cosine` 时启用 `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=epochs, eta_min=lr_min)`
    - histories 追加列：`lr`
    - summary.json 追加字段：`lr_scheduler`、`lr_min`
  - `scripts/train_sota_stage_survival.py`
    - 透传参数：`--lr_scheduler`、`--lr_min`、`--selection_min_epochs`、`--early_stop_patience`

### 2.2 GDC 下载脚本 DX-only 强保证（历史改动）

已对 `scripts/download_brca200_supplement_gdcapi.py` 做过“机制级强保证”增强（require DX、clinical event 过滤、dry-run 先输出 manifest 等），用于避免无效下载与数据污染。

## 3. 论文数值：已校准的关键表（最终以 summary.json 为准）

### 3.1 batch=16 不同窗口（bs16，Stage2 best_test_c_index）

说明：以下为“纠错版”（基于本机 `outputs/**/stage2/summary.json` 汇总）。UCEC 部分窗口在本机缺失则留空。

```text
缩写	癌种全称	病例数（Case）	twostage_fixed_off_bs16	twostage_param_best_bs16	twostage_param_safe_bs16	twostage_onek_bs16	twostage_twok_bs16
LUAD	肺腺癌	403	0.694241 ± 0.057097	0.704798 ± 0.054171	0.703895 ± 0.055623	0.698411 ± 0.057530	0.694221 ± 0.053539
BRCA	乳腺浸润癌	642	0.702507 ± 0.063322	0.707273 ± 0.058858	0.708276 ± 0.061677	0.704482 ± 0.064666	0.708792 ± 0.064604
LUSC	肺鳞癌	220	0.627590 ± 0.084020	0.635491 ± 0.079745	0.633279 ± 0.082538	0.626934 ± 0.088450	0.627500 ± 0.084691
PAAD	胰腺癌	100	0.642290 ± 0.085190	0.633779 ± 0.081648	0.629139 ± 0.079562	0.643415 ± 0.073996	0.643201 ± 0.077167
UCEC	子宫内膜癌	387			0.681101 ± 0.057642		
BLCA	膀胱尿路上皮癌	312	0.677753 ± 0.055287	0.673352 ± 0.051783	0.677650 ± 0.050343	0.677895 ± 0.051430	0.677895 ± 0.051430
```

### 3.2 batch=8/16/32/64（param_safe）

说明：以下为当时给出的“纠错版”（Stage2 best_test_c_index）。部分 cohort 的 bs64 缺失则留空。

```text
缩写	癌种全称	病例数（Case）	batch=8（std）	batch=16（std）	batch=32（std）	batch=64（std）
LUAD	肺腺癌	403	0.712449 ± 0.050395	0.703895 ± 0.055623	0.700839 ± 0.058491	0.671987 ± 0.065888
BRCA	乳腺浸润癌	642	0.720657 ± 0.062696	0.708276 ± 0.061677	0.683059 ± 0.063798	0.670511 ± 0.055266
LUSC	肺鳞癌	220	0.705910 ± 0.057184	0.633279 ± 0.082538	0.592732 ± 0.059808	0.534568 ± 0.055794
PAAD	胰腺癌	100	0.688023 ± 0.085928	0.629139 ± 0.079562	0.634826 ± 0.077207	0.587201 ± 0.083211
UCEC	子宫内膜癌	387	0.688932 ± 0.060096	0.681101 ± 0.057642	0.675292 ± 0.065532	
BLCA	膀胱尿路上皮癌	312	0.697897 ± 0.043070	0.677650 ± 0.050343	0.658296 ± 0.046870	0.614148 ± 0.020514
```

## 4. bs64 对齐参考论文训练设置：实验状态与阶段性结论

目标：
- bs64 在 cosine + wd=1e-5 下能否“调回来”
- 调回来后，窗口机制（param_safe）是否仍有 pp 级增益（paired Δ）

对齐参考论文文字：`lr=1e-4 + cosine decay, AdamW, wd=1e-5, epochs=20, batch=64, Cox loss`。

当前产物：
- Stage1（完成 15/15）：`outputs/BRCA_pretrain_bs64_ep20_cosine_wd1e5/BRCA/seed*/fold*/stage1/summary.json`
- Stage2（进行中，完成 3/15）：  
  - fixed_off：`outputs/BRCA_fixed_off_bs64_ep20_cosine_wd1e5/BRCA/.../stage2/summary.json`
  - param_safe：`outputs/BRCA_param_safe_bs64_ep20_cosine_wd1e5/BRCA/.../stage2/summary.json`

阶段性统计（n=3）：
- fixed_off：0.675164 ± 0.067315
- param_safe：0.677224 ± 0.065869
- paired Δ（param_safe - fixed_off）：+0.002060 ± 0.002914

同 seed0、fold0-2 的逐 fold 对照（Stage2 best_test_c_index）：

```text
seed	fold	old_bs8	old_bs16	old_bs32	old_bs64	new_cos64_fixed_off	new_cos64_param_safe	new-param_safe - old_bs64	new-param_safe - new-fixed_off
seed0	fold0	0.683379	0.706731	0.749313	0.710165	0.626374	0.632555	-0.077610	+0.006181
seed0	fold1	0.718022	0.733940	0.690165	0.640136	0.628766	0.628766	-0.011370	+0.000000
seed0	fold2	0.759100	0.746525	0.733289	0.664461	0.770351	0.770351	+0.105890	+0.000000
```

注意：当前仅 3/15，方差很大，必须等 15/15 齐再下定论。

## 5. 数据清理（已执行）

删除前大小：
- `data/downloads`：1.1T
- `data/raw_svs`：2.2T

删除依据（只对“当前 active phase2 splits + DX(含UUID)”做硬核验）：
- 在 active splits（BRCA/LUAD/LUSC/PAAD/UCEC/BLCA）中，共 2064 个 `slide_id`
- 其中 DX 且含 UUID 的 `slide_id` 共 1754 个
- 检查这些 `slide_id` 均存在对应 uni1024 特征 `.h5`（missing=0）

已永久删除：
- `data/downloads`
- `data/raw_svs`
- `data/gdc.zip`（原文件仅 9B）

当前 `data/` 大小约：153G（不含 raw svs/downloads）

## 6. 迁移到新服务器（不含 raw svs 的最小可复现资产）

迁移目录（建议）：
- `outputs/`
- `data/splits/`
- `data/wsi_features/`（uni1024 特征）
- `data/raw_rna/`（如训练需要）

rsync 示例（从旧机推到新机）：

```bash
rsync -avP --partial --inplace \
  -e "ssh -p 15051 -o StrictHostKeyChecking=no" \
  /root/autodl-tmp/R2wsp/_migration_pack/ \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/R2wsp/_migration_pack/
```

## 7. GitHub 分支

- 目录：`/root/autodl-tmp/mmp-test/R2wsp`
- 分支：`exp/r2wsp-test`
- 远端：`git@github.com:lillian0406/R2wsp.git`
- 状态：`git push origin exp/r2wsp-test` 返回 `Everything up-to-date`（远端已是最新）

## 8. 后续建议（最小成本闭环）

1) 先把 bs64 对齐实验 stage2 跑齐到 15/15，再输出可写论文的结论：
   - new cos bs64 vs old bs64 的 mean±std（及可选 paired）
   - param_safe vs fixed_off 的 paired Δ（pp 级增益是否稳定）
2) 若仍不稳定，优先检查：
   - best_epoch 分布是否集中在极早期（如 epoch=1），避免“早期波动”误导结论
   - 是否需要按 step 对齐（增加 epochs 或调整 early stop 口径）

