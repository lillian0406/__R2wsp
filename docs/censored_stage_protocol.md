# Censored-Stage Survival Protocol（删失分两阶段 5 折协议）

本说明对应脚本：

- 数据拆分：[build_censored_stage_splits.py](file:///root/autodl-tmp/R2wsp/scripts/build_censored_stage_splits.py)
- 两阶段训练：[train_censored_stage_survival.py](file:///root/autodl-tmp/R2wsp/scripts/train_censored_stage_survival.py)

> 严格遵循最终口径：
> - **WSI / RNA 两条线都使用最原始、无 geo 的基线配置**（不启用任何 curve / b_points / tile geo）；
> - 不修改主线 [train_direct_wsi_rna_survival.py](file:///root/autodl-tmp/R2wsp/scripts/train_direct_wsi_rna_survival.py)；
> - 训练目标：先在“纯无删失子数据”上学会排序（Phase-1），再在“删失 5 折均分 + 小窗口损失调参”下微调（Phase-2），不依赖任何 selection_metric 的最佳 checkpoint 做最终报告，直接用 final.pt。

## 1. 目标

### Phase-1：纯无删失初始化训练
- 对官方 split 的每一个外折（k=0..4），把 `train.csv` 和 `test.csv` 都先按 `dss_censorship < 0.5` 过滤，得到“纯无删失子数据集”；
- 在 5 个官折上分别跑 **3 个 seed**，总共 5×3 = 15 次训练，保存每个 run 的 `best.pt`（按 `val_c_index_ema` 选）；
- 预期：任务难度下降，性能会比主线略高，大概 **~0.70** 左右；
- 目的：给 Phase-2 一个“已经学会正确排序”的初始 weights，避免 Phase-2 因为删失混进来而早期走歪。

### Phase-2：独立重分 5 大折 + 删失 5 份一一对应 + 窗口化 Cox 调参
- **先对全局 pool（官折 k=0..4 的 train+test 合并去重）重新做独立拆分**：
  - 把「无删失 cases」独立重分 5 折 `u_0..u_4`；
  - 把「删失 cases」独立重分 5 份不重复 `c_0..c_4`（均分）；
  - 第 i 个大折：
    - **train = 其余 4 折的 uncensored ∪ 其余 4 份的 censored**（即每折 train 里 uncensored 占 4/5，censored 占 4/5，保证对应 1:1 对应难度一致）；
    - **test = uncensored_i ∪ censored_i**（默认口径，和训练任务同分布，默认走这套）；
    - **或 test = uncensored_i 纯无删失**（对照口径，旧设计，保留做对比）；
- 对每个大折 i ∈ {0..4}，跑 **3 个 seed**，共 5×3 = 15 次训练；
- 初始化 checkpoint：取 Phase-1 官折 i 同 seed 下的 `best.pt`（5×3 刚好一一对应，避免混 seed）；
- 损失函数采用“窗口软开关”：
  - f(x) = A · σ(k(x − L)) · σ(−k(x − U))
  - 参数：A 是幅度系数（灵敏度），默认 A=0.99；L/U 是窗口上下界；k 是陡峭系数（进出门速度）
  - 当 val_c_index_ema ∈ 窗口：f ≈ A = 0.99，L_total = 0.99 × L_cox（相当于窗口内 Cox 最大削弱 1%）
  - 当 val_c_index_ema 在窗口外：f ≈ 0，再取 max(w, loss_window_eps) 避免梯度完全归零；
  - 窗口权重基于“上一 epoch 结束的 val_c_index_ema”计算（避免未来信息泄漏）；
- checkpoint 产物：
  - Phase-2 每轮训练结束覆盖写出 `final.pt`（**不做 best-val select**），直接用最后一轮的权重作为 final checkpoint；
  - best.pt 也同时保存（仅调试/ablation 对照用，**最终报告不用**）；
- 预期：Phase-2 final train 端 C-index ~0.69，验证/测试端 ~0.67（原预期）。

> 说明：窗口作用区间是 0.68-0.72，即 L=0, U=1”，这两者在字面上冲突，本脚本默认实际窗口按 **L=0.68, U=0.72** 生效；`L=0, U=1` 等价于全区间触发（会让 A 变成全局系数而不再是窗口门控），如要做这个对照，只需在命令行把 `--loss_window_lower 0 --loss_window_upper 1` 覆盖即可。

## 2. 固定基线（no-geo guardrails）

所有 Phase-1 / Phase-2 训练都会强制启用下面硬约束（如不满足直接报错）：

| 项目 | 固定值 | 原因 |
|---|---|---|
| `model_variant` | `baseline` / `gate_only` | `custom` 路径包含 geo 分支逻辑，为避免静默启用，直接禁用 |
| `wsi_geo_type` | `none` | 严格无 geo |
| `rna_geo_type` | `none` | 严格无 geo |
| `cross_modal_fusion` | 默认 `concat` | 最原始融合（主线无 geo 对照） |
| `use_multi_slide` | 默认 `false`（可覆盖） | 如要 heavy joint 版本，可手动开 `--use_multi_slide`，但脚本依然会强制 no-geo |

## 3. 阶段数据流图

```text
[官方 split (TCGA_LUAD_overall_survival_k=0..4)]
        │
        ▼
[build_censored_stage_splits.py]  (Phase-0 离线数据拆分)
        │
        ├─► phase1_outer5/fold_i/train.csv    # 纯无删失 train（原官折 train 过滤后）
        │       phase1_outer5/fold_i/test.csv     # 纯无删失 test（原官折 test 过滤后）
        │
        ▼ Phase-1（官折 5 × seed 3 = 15 次） stage=1，不调损失，只存 best.pt
        [train_censored_stage_survival.py --stage 1]
        └─► outputs/phase1/outer_{i}_seed{s}/best.pt  (phase1_best 共 15 个)
        │
        ▼ Phase-2（独立新 5 大折 × seed 3 = 15 次） stage=2，加载同 i 的 phase1 best.pt 继续训练
        phase2_independent5/fold_i/
          ├─ eval_with_censored/{train.csv, test.csv}      # 默认：test 含删失（和训练同口径，现在更倾向的设计）
          └─ eval_uncensored_only/{train.csv, test.csv}    # 对照：test 纯无删失（旧设计）
        │
        ▼ stage=2 + f(x)=A·σ·σ 窗口化 Cox
        [train_censored_stage_survival.py --stage 2 --pretrain_checkpoint phase1/outer_i_seed{s}/best.pt]
        └─► outputs/phase2/fold_{i}_seed{s}/
               ├─ best.pt    (仅调试对照用，最终报告不用)
               └─ final.pt   (每轮末尾覆盖写，**最终 checkpoint，用于验证**)
```

## 4. 产物目录结构

```text
/root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol
  ├─ phase1_outer5/
  │    ├─ fold_0/{train.csv, test.csv, manifest.json}
  │    └─ ... fold_4
  └─ phase2_independent5/
       ├─ manifest.json (phase2 独立 5 折的 uncensored/censored 分配总表)
       └─ fold_i/
            ├─ eval_with_censored/{train.csv, test.csv, manifest.json}   (默认)
            └─ eval_uncensored_only/{train.csv, test.csv, manifest.json} (对照)

/root/autodl-tmp/R2wsp/outputs (可改)
  └─ censored_stage_v2/
       ├─ phase1/outer_0_seed0/{best.pt, summary.json, histories/seed0.csv}
       ├─ ... outer_4_seed2
       └─ phase2/
            ├─ fold_0_seed0/{best.pt, final.pt, summary.json, histories/seed0.csv}
            └─ ... fold_4_seed2
```

每个 manifest.json 都明确记录：
- phase2 每折 uncensored/censored case ID（互相不重复，sum = 总数）；
- 各折 case 数量；
- 方便后续回溯“这个 final.pt 是哪一折训出来的”。

## 5. 命令模板

先建 Phase-0 数据（每次协议改完都要重新跑一次，因为 Phase-2 的拆分逻辑变了）：
```bash
/root/autodl-tmp/venvs/tcga1126/bin/python scripts/build_censored_stage_splits.py
```

Phase-1（官折 5 × seed 0/1/2 = 15 次）：
```bash
FOLD=0
SEED=0
/root/autodl-tmp/venvs/tcga1126/bin/python scripts/train_censored_stage_survival.py \
  --split_dir /root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol/phase1_outer5/fold_${FOLD} \
  --wsi_feature_source plip_luad_256 \
  --rna_mode vec \
  --model_variant baseline \
  --cross_modal_fusion concat \
  --batch_size 16 --num_workers 2 --pin_memory \
  --epochs 24 --seed ${SEED} --stage 1 \
  --out_dir /root/autodl-tmp/R2wsp/outputs/censored_stage_v2/phase1/outer_${FOLD}_seed${SEED}
```

Phase-2（独立 5 大折 × seed 0/1/2 = 15 次；**默认 eval_with_censored 口径**）：
```bash
FOLD=0
SEED=0
EVAL_KIND=eval_with_censored   # 或者改成 eval_uncensored_only 做对照
/root/autodl-tmp/venvs/tcga1126/bin/python scripts/train_censored_stage_survival.py \
  --split_dir /root/autodl-tmp/R2wsp/data/splits/censored_stage_protocol/phase2_independent5/fold_${FOLD}/${EVAL_KIND} \
  --wsi_feature_source plip_luad_256 \
  --rna_mode vec \
  --model_variant baseline \
  --cross_modal_fusion concat \
  --batch_size 16 --num_workers 2 --pin_memory \
  --epochs 24 --seed ${SEED} \
  --stage 2 \
  --pretrain_checkpoint /root/autodl-tmp/R2wsp/outputs/censored_stage_v2/phase1/outer_${FOLD}_seed${SEED}/best.pt \
  --loss_window_lower 0.68 \
  --loss_window_upper 0.72 \
  --loss_window_k 50.0 \
  --loss_window_A 0.99 \
  --loss_window_eps 1e-3 \
  --out_dir /root/autodl-tmp/R2wsp/outputs/censored_stage_v2/phase2/${EVAL_KIND}/fold_${FOLD}_seed${SEED}
```
*想跑 A=1（只靠窗口门控、不额外缩放 Cox）的对照，只需把 `--loss_window_A 1.0` 覆盖即可。*
*想跑“全区间窗口 L=0 U=1”的 ablation，再加 `--loss_window_lower 0 --loss_window_upper 1`。*

## 6. 验证（Phase-2 final.pt 直接跑）

报告最终结果的**正确口径**是：
- 对每个 `fold_i / seed_s`，拿 `final.pt` 在对应 split 的 `test.csv` 上评估；
  - 如果跑的是 `eval_with_censored`：test 里 censored + uncensored 都参与 C-index 计算（C-index 本来就支持 censorship，不会错）；
  - 如果跑的是 `eval_uncensored_only`：test 是纯无删失（对照组）；
- 最后把 15 个 final 的 test C-index 汇总（均值 / 标准差 / 每折 / 每 seed），做 Phase-2 最真实效果；
- best.pt 的 numbers 可以同步记录但**不用于主表**（主表必须按 final.pt）。

脚本里 history 最后一行的 `test_c_index` 已经和主线 `evaluate()` 等价，可直接用 `summary.json` 取 numbers。

## 7. 当前已经验证过的事情

- [x] Phase-0 拆分脚本语法正确；
- [x] 训练脚本语法正确（已 `py_compile` 验证）；
- [x] No-geo 硬护栏：脚本显式拒绝 custom / any-geo，避免静默污染；
- [x] final.pt 每轮覆盖写，Phase-2 可直接按“最后一轮”拿 checkpoint；
- [x] 窗口公式按修正的 f(x) = A·σ(k(x-L))·σ(-k(x-U)) 实现；
- [x] 窗口权重基于上一 epoch val_c_index_ema（避免未来信息泄漏），窗口外仍保留 `loss_window_eps` 最小权重防死梯度；
- [ ] Phase-0 新拆分产物还没实际跑；
- [ ] Phase-1 15 次实际训练待跑；
- [ ] Phase-2 15 次实际训练待跑。

## 8. 可调的关键超参

这些超参在 Phase-2 可以先按默认跑，等有结果后再邻域精扫：

| 超参 | 含义 | 默认 | 下一步建议扫的范围 |
|---|---|---|---|
| `loss_window_lower` | L：窗口下界 | 0.68 | 0.50 / 0.60 / 0.66 / 0.68 / 0.70 |
| `loss_window_upper` | U：窗口上界 | 0.72 | 0.72 / 0.74 / 0.75 / 0.80 |
| `loss_window_k` | k：窗口陡峭度（越大越接近硬窗，“进出门的速度”） | 50.0 | 10 / 20 / 50 / 100 |
| `loss_window_A` | A：窗口幅度系数（“灵敏度”，窗口内最大值） | 0.99 | 0.90 / 0.95 / 0.99 / 1.00（A=1 只靠门控开关，不额外缩放 Cox） |
| `loss_window_eps` | 窗口外最小权重（防梯度归零） | 1e-3 | 1e-4 / 1e-3 / 1e-2 / 0.05 |
| `eval_kind` | val/test 口径（不是超参，是拆分物理文件层面对照） | eval_with_censored | eval_with_censored vs eval_uncensored_only 两套都跑 |

---

## 9. 历史勘误 / 边界说明

- L=0, U=1按字面等于“整个 [0,1] 都是窗口，f(x) 全是 A”，那就变成纯 Cox 乘 A 的全局缩放，不再是“只有中间才调节”。把窗口按 0.68-0.72 实现，如还要做 0~1 作为 ablation，覆盖命令行参数即可；
- 只覆盖写、不做任何 select**；best.pt 也同时保存方便后续对照，但最终报告请按 final.pt 的 numbers 汇报；
- 删失平等划分、每个 fold 稍微小一点难度，本协议 Phase-2 的 split 是把 **uncensored 和 censored 都独立分 5 折一一对应**，每折 train = 其他 4 份，严格满足“每个 fold 删失难度近似一致”；
- **第 4 点纠正的训练量**：Phase-2 不再是“嵌套 25”，而是“独立 5 大折 × 3 seed = 15 次”，Phase-1 也同步跑 5×3=15 次，保证 seed 一一映射。
