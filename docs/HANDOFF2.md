# 项目交接说明（R2wsp｜论文写作向）

更新时间：2026-09-02（Asia/Shanghai）

## 1. 本交接文档的范围

- 本文档用于“论文方法写作/口径统一”，不覆盖具体实验跑法与数据清理细节（实验与数据分析在其他对话窗进行）。
- 当前论文方法最终口径：two-stage + pullback + validation-driven window scheduler（param-safe）。

## 2. 最终方法口径（写论文用）

### 2.1 问题形式化（Cox 排序学习）

- 数据：\(\mathcal{D}=\{(x_i,t_i,\delta_i)\}_{i=1}^n\)，\(\delta_i=1\) 为未删失事件。
- 风险函数：\(r_\theta(x_i)\in\mathbb{R}\)。
- 任务损失：\(\mathcal{L}_{\mathrm{cox}}(\theta;\mathcal{D})\) 为 Cox-NPLL（风险集上的排序学习目标）。

### 2.2 Two-stage + pullback（与实现一致）

- Stage-1（uncensored pretraining）：
  \[
  \theta^{(1)}=\arg\min_\theta \mathcal{L}_{\mathrm{cox}}(\theta;\mathcal{D}_u),\quad \mathcal{D}_u=\{(x_i,t_i,1)\}\subset\mathcal{D}.
  \]
- Stage-2（finetuning on all cases + pullback）：
  \[
  \mathcal{R}(\theta;\theta^{(1)})=\|\theta-\theta^{(1)}\|_2^2,\quad \mathcal{L}_{\mathrm{pull}}(\theta)=\lambda\mathcal{R}(\theta;\theta^{(1)}).
  \]
- Stage-2（epoch 级门控目标）：
  \[
  \mathcal{L}^{(e)}(\theta)=w_e\cdot \mathcal{L}_{\mathrm{cox}}(\theta;\mathcal{D})+(1-w_e)\cdot \mathcal{L}_{\mathrm{pull}}(\theta).
  \]

注意：正文不引入/不提 geo 正则（不作为论文方法的一部分）。

### 2.3 Param-Safe Window Scheduler（最终采用）

- 验证信号：\(m_e\) 使用验证集 C-index 的 EMA（无未来信息泄漏：第 \(e\) 轮训练用 \(m_{e-1}\) 生成 \(w_e\)，且 \(w_1=1\)）。
- 双侧 sigmoid 软窗口：
  \[
  \tilde{w}_e=A\cdot \sigma(k_e(m_{e-1}-L_e))\cdot \sigma(k_e(U_e-m_{e-1})),\quad A\in(0,1].
  \]
  \[
  w_e=\max(\tilde{w}_e,\varepsilon).
  \]
- Param-safe 参数化（center/width/transition）：
  \[
  L_e=c_e-\frac{\omega_e}{2},\quad U_e=c_e+\frac{\omega_e}{2}.
  \]
  - center：\(c_e\) 取历史验证信号 EMA（避免 best 追峰导致权重塌陷/冻结）。
  - width：\(\omega_e\) 取历史分布中心分位宽度（central quantile width），\(q=0.8\)；并施加下限 \(\omega_{\min}=0.06\)（防过窄窗口）。
  - transition：用过渡宽度 \(\delta\) 控制边界软硬，\(\delta=0.25\)，并由 logistic 几何关系映射到斜率：
    \[
    k_e \approx \frac{2.944}{\delta}.
    \]
  - floor：\(\varepsilon=0.05\)（防止任务梯度消失导致训练冻结）。

## 3. 论文写作产物（可直接复用）

### 3.1 ICLR 风格 Abstract / Intro / Contributions

- 已在对话中给出 “ICLR 风格英文 Abstract + Intro 前两段 + 3 条贡献”版本，口径为 two-stage + param-safe，不引入编码器对比贡献点。

### 3.2 Related Work（ICLR 风格重构）

- 按方法类别分段（fusion / censoring-aware objective / regularization & curriculum / RNA features / evaluation protocol）。
- 每段末尾一句 “Difference to ours”，避免情绪化对比与硬塞数值表。

### 3.3 Method 章节写法

- 4.1–4.4 的推荐写法中，4.4（Param-safe 窗口权重）已生成 ICLR 风格中文整段，且修正了常见符号/公式错误：
  - 区分 \(\tilde{w}_e\) 与 \(w_e\)
  - \(\omega_{\min}\) 约束的是窗口宽度 \(\omega_e\)，不是权重 \(w_e\)
  - \(k\approx 2.944/\delta\)（不是乘）

### 3.4 Algorithm 1（建议保留）

建议保留 Algorithm box，用于明确：
- two-stage 流程
- “无未来信息泄漏”：用 \(m_{e-1}\) 生成 \(w_e\)
- epoch 结束后更新历史并生成下一轮 \(w_{e+1}\)

## 4. 工程对应关系（写论文时避免写错）

### 4.1 论文方法的最终入口脚本（wrapper）

- 统一入口：[train_sota_stage_survival.py](file:///root/autodl-tmp/R2wsp/scripts/train_sota_stage_survival.py)
- 固定输入协议：
  - `--wsi_feature_source uni1024`
  - `--rna_mode omics` + hallmark gene sets + gencode gtf
  - multi-slide：`--use_multi_slide --multi_slide_mode slide_attn_case_attn --multi_slide_tile_budget_mode case_shared`
- Stage2 的 `--window param_safe` 会翻译成一组 `--loss_window_*` 参数（center=ema、width=quantile、transition、eps、width_min 等）。

### 4.2 真正执行损失与门控的位置（实现依据）

- 训练主体：[train_censored_stage_survival.py](file:///root/autodl-tmp/R2wsp/scripts/train_censored_stage_survival.py)
- Stage-2 的门控损失形式与实现一致：
  - `loss = w * raw_loss + (1-w) * pull_loss`
  - 其中 `raw_loss` 为 Cox-NPLL（论文不引入 geo）
  - pullback：`pullback_lambda * ||θ-θ_stage1||^2`
  - `w_e` 由上一轮验证信号产生，并做 `max(w, eps)`

## 5. 口径与风险点（论文写法必读）

### 5.1 “no-window baseline” 的定义要与实现一致

- 当前实现中，若设置 \(w\equiv 1\)，则 \((1-w)=0\)，pullback 会被关闭。
- 因此写论文时需清晰区分：
  - “关窗但保留 pullback” vs “完全不门控且不回拉”
  - ablation 的命令与公式要对齐（避免被审稿人抓实现不一致）。

### 5.2 常数解释口径（避免看起来拍脑袋）

- 0.05/0.95：用于定义 sigmoid “near-off/near-on” 的约定阈值（不是额外调参）。
- 2.944：logistic 几何常数，用于把过渡宽度 \(\delta\) 映射到斜率 \(k\)。
- \(\varepsilon=0.05\)、\(\omega_{\min}=0.06\)、\(q=0.8\)、\(\delta=0.25\)：作为 design constants（全队列共享、非逐队列调参），对应防冻结、防窄窗、鲁棒尺度估计、边界软硬控制。

## 6. BRCA_supp300 合并到 BRCA 的注意事项（只写口径，不做实验步骤）

- 不要把 `UNI_todo_stems_BRCA_OFFICIAL_GDC.txt` 这类 “case 级短 ID” 合并进 slide-level todo manifest，否则会大量 `missing_raw_svs`。
- 合并 slide-level manifest 的推荐方式：仅 union
  - `UNI_todo_stems_BRCA.txt`
  - `UNI_todo_stems_BRCA_supp300.txt`
- “remaining=1” 的常见原因之一：某些 slide 会出现 `no tissue tiles selected`（应在数据/预处理层处理，而非强行写成方法贡献）。

