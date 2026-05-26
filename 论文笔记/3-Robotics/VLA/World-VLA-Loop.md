---
title: "World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy"
method_name: "World-VLA-Loop"
authors: [Xiaokang Liu, Zechen Bai, Hai Ci, Kevin Yuchen Ma, Mike Zheng Shou]
year: 2026
venue: arXiv
tags: [vla, world-model, video-diffusion, reinforcement-learning, robot-manipulation, closed-loop, reward-prediction]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2602.06508v2
created: 2026-05-26
---

# 论文笔记：World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore (Show Lab) |
| 日期 | February 2026 (v2: May 2026) |
| 项目主页 | N/A |
| 对比基线 | [[Cosmos-Policy\|Cosmos-Predict 2]], [[GRPO]], SimpleVLA-RL |
| 链接 | [arXiv](https://arxiv.org/abs/2602.06508) / [PDF](https://arxiv.org/pdf/2602.06508) / Code: N/A |

---

## 一句话总结

> World-VLA-Loop 通过 **SANS 数据集** + **状态感知视频世界模型** + **策略-模拟器闭环协同进化**，实现了在视频世界模型内进行 [[VLA]] 强化学习后训练，真机两次迭代后成功率提升 36.7% 与 26.6%。

---

## 核心贡献

1. **SANS（Success and Near-Success）数据集**: 首次提出在视频[[World Model|世界模型]]训练中混合成功与"近似成功"失败轨迹，强制模型学习细粒度物理动力学，解决现有视频世界模型对错误动作产生幻觉式成功的问题。
2. **状态感知视频世界模拟器**: 在 [[DiT]] 扩散主干上集成轻量级[[预测编码头|奖励预测头]]，联合监督视频流匹配损失与二元奖励 MSE 损失，使世界模型同时承担"环境模拟"与"奖励信号"双重角色，无需外部 VLM 评判。
3. **闭环协同进化范式（World-VLA-Loop）**: 提出策略 RL 训练与世界模型再训练的迭代循环——策略 rollout 反哺 [[SANS 数据集]]，更新后的世界模型支撑下一轮 [[GRPO]] 优化，在 LIBERO 与真机 Franka 平台均显著优于 SFT 基线。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型的强化学习后训练（RL post-training）严重依赖与真实环境的交互，但物理机器人交互成本高昂、数据效率低下。现有方案试图用[[潜空间世界模型|视频世界模型]]作为虚拟环境替代，但面临两大障碍：

1. **动作跟随不准**: [[Cosmos-Policy|Cosmos-Predict 2]] 等模型在面对错误动作时常幻觉式地生成成功结果，反映了对细粒度物理动力学的弱[[Embodied AI|具身]]接地能力
2. **缺失原生奖励**: 视频世界模型只生成像素，没有内建奖励信号，难以驱动 [[GRPO]] 等需要标量回报的 RL 算法

### 现有方法的局限

- **3D 仿真重建**（如 [[ManiSkill]]）: 需要昂贵的 mesh 建模与物理引擎适配，[[sim-to-real]] 鸿沟显著
- **纯视频世界模型**（如 [[UniSim]]、[[IRASim]]、[[Vid2World]]）: 仅在成功轨迹上训练，无法学习"差之毫厘谬以千里"的接触动力学
- **外置 VLM 奖励**（如 [[Qwen3-VL]]-8B 作 reward critic）: 与世界模型解耦，导致视觉-奖励对齐性差，本文实验中比集成奖励头低 6%-32%
- **开环 RL 后训练**: 策略改进后世界模型未更新，分布漂移使模拟环境与策略行为越来越偏离训练分布

### 本文的动机

作者观察到：策略在真实环境中失败时往往是**末端执行器位置存在微小误差**导致抓取失败这类"near-success"案例。如果世界模型只见过完美成功轨迹，就无法对这类边界情况建模，反而会奖励错误动作。因此需要：

1. 在训练数据层面注入"近似成功"的失败样本（[[SANS 数据集]]）
2. 在模型层面让奖励预测与视频生成共享[[DiT]] 潜表征
3. 在系统层面让世界模型与策略形成闭环共进化

---

## 方法详解

### 模型架构

World-VLA-Loop 由三大组件构成闭环系统：

- **输入**: 语言指令 $l$ + 初始观测 $o_0$ + 动作序列 $\{a_t\}_{t=0}^{T-1}$（6-DoF [[末端执行器]]位姿 + 夹爪开合）
- **Backbone**: [[Cosmos-Policy|Cosmos-Predict 2]]（基于 [[DiT]] 的视频扩散 Transformer）
- **核心模块**:
  - **动作嵌入**: 6-DoF 位姿 + 夹爪状态 → 潜张量，按加法注入 [[DiT]] 的时间步嵌入
  - **[[预测编码头|奖励预测头]]** $\phi$: 轻量 [[MLP]]，将最终去噪潜变量 $z_t$ 映射为标量奖励 $\hat{r}_t$
  - **[[SANS 数据集]]**: 成功 + 近似成功失败轨迹混合
- **输出**: 未来 24 帧视频 $\hat{o}_{1:T}$ + 每步标量奖励 $\hat{r}_{1:T}$
- **统一 chunk size**: 24 帧（所有任务）

### 核心模块

#### 模块 1: [[SANS 数据集]]（Success and Near-Success Dataset）

**设计动机**: 让世界模型学会区分"差一点点就成功"与"明显失败"这类细粒度接触动力学差异。

**具体实现**:
- **来源 A — 人工遥操作成功轨迹**: 通过 teleoperation 收集标准成功示范
- **来源 B — 策略 rollout 近似失败**: SFT 基线策略实际部署时的失败案例，主要表现为末端位置微小偏差导致抓取失败
- **规模**:
  - **ManiSkill 预训练池**: 35k 视频-动作对，覆盖 23 个任务
  - **下游任务集**: 每任务约 50 条成功 + 50 条失败（LIBERO / 真机）
- **二元奖励标签**: 每条轨迹的末帧标注 0/1，作为奖励预测头的监督信号

#### 模块 2: [[状态感知视频世界模拟器]]（State-aware Video World Simulator）

**设计动机**: 让世界模型既能生成动作条件视频，又能预测奖励，避免外置 critic 与生成器之间的感知-奖励一致性脱节。

**具体实现**:
- **视频生成分支**: 沿用 [[Cosmos-Policy|Cosmos-Predict 2]] 的 [[Flow Matching]] 目标 $\mathcal{L}_{flow}$，自回归预测未来视频帧
- **动作条件化**: 动作潜张量与扩散时间步嵌入相加，注入到每个 [[DiT]] 块
- **奖励预测分支**: 最终去噪步的潜变量 $z_t$ 经 [[MLP]] $\phi$ 输出 $\hat{r}_t$
- **联合训练**: 加权混合视频损失与奖励 MSE 损失，权重 $\lambda$ 按 [[EDM 噪声加权]] 框架按噪声水平调制
- **关键设计**: 奖励头共享视频生成的潜表征，迫使去噪过程同时编码视觉与任务完成度，形成隐式互监督

#### 模块 3: 世界模型驱动的策略 RL 后训练（World Simulator for Policy RL Post-Training）

**设计动机**: 将训练好的世界模型当作虚拟 Gym 环境，使用 [[GRPO]] 优化 VLA 策略，无需真机交互。

**具体实现**（4 阶段流水线）:
1. **阶段 1 — SANS 构建**: 通过遥操作 + 当前策略 rollout 构建混合数据集
2. **阶段 2 — 世界模型预训练**: 在 ManiSkill SANS 上联合学习视频与奖励
3. **阶段 3 — 任务微调 + 策略 RL**:
   - 用任务专属的 ∼100 条 SANS 微调世界模型
   - 在世界模型内 rollout VLA 策略
   - 用世界模型预测的奖励 $\hat{r}_t$ 驱动 [[GRPO]]（继承 SimpleVLA-RL 流水线）
4. **阶段 4 — 闭环增强**: 优化后的策略再次在真实/模拟环境 rollout，新失败案例并入 SANS，回到阶段 3 迭代

> **闭环本质**: "refined world model supports iterative VLA post-training inside the virtual environment, while rollouts from each updated policy are fed back to augment training and fine-tune the world model."

---

## 关键公式

### 公式 1: [[联合训练目标|状态感知联合损失]]

$$
\mathcal{L} = \mathcal{L}_{flow} + \lambda \sum_{t=1}^{T} \big\| \hat{r}_t - r_t \big\|^2
$$

**含义**: 世界模型的总损失由两部分组成——视频生成的 [[Flow Matching]] 损失 + 整条轨迹上的奖励预测 MSE 损失。两者共享 [[DiT]] 主干的潜表征，强制视觉去噪过程同时编码任务完成度。

**符号说明**:
- $\mathcal{L}_{flow}$: [[Flow Matching]] 损失（Lipman et al. 2022），监督视频帧预测
- $\hat{r}_t \in \mathbb{R}$: 时间步 $t$ 的预测奖励（标量）
- $r_t \in \{0, 1\}$: 时间步 $t$ 的真值二元奖励（成功/失败）
- $T$: 视频 chunk 长度（本文 $T=24$）
- $\lambda$: 奖励损失权重，按噪声水平调制（[[EDM 噪声加权]] 框架）

### 公式 2: 奖励预测头

$$
\hat{r}_t = \phi(z_t)
$$

**含义**: 奖励预测头 $\phi$ 是一个轻量 [[MLP]]，将每帧最终去噪步的潜变量映射为标量奖励，作为 [[GRPO]] 的回报信号。

**符号说明**:
- $z_t \in \mathbb{R}^{H \times W \times C}$: 第 $t$ 帧在最终去噪步的 [[DiT]] 潜表征
- $\phi: \mathbb{R}^{H \times W \times C} \to \mathbb{R}$: 轻量 [[MLP]] 奖励头
- $\hat{r}_t$: 时间步 $t$ 的预测奖励

### 公式 3: 噪声依赖的奖励权重（[[EDM 噪声加权]]）

$$
\lambda(\sigma) = \frac{1}{\sigma^2 + \sigma_{data}^2}
$$

**含义**: 借鉴 [[EDM 噪声加权|EDM 框架]]，奖励损失权重按当前去噪步的噪声水平 $\sigma$ 调制——噪声小（接近真实图像）时给奖励项更大权重，噪声大时主要让视频生成主导训练。

**符号说明**:
- $\sigma$: 当前去噪步的高斯噪声标准差
- $\sigma_{data}$: 数据分布的标准差（[[EDM 噪声加权|EDM]] 中通常取 0.5）
- $\lambda(\sigma)$: 奖励损失在该噪声水平的权重

---

## 关键图表

### Figure 1: World-VLA-Loop 范式对比与真机收益

![Figure 1](https://arxiv.org/html/2602.06508v2/x1.png)

**说明**:
- **(a)** 三种世界模型 VLA RL 范式对比：
  - **3D World Reconstruction**（如 ManiSkill）: 需要重建完整 3D 环境，部署成本高
  - **Pure Video Simulator**（如 IRASim）: 仅视频生成，无奖励信号
  - **World-VLA-Loop（本文）**: 通过 [[SANS 数据集]] + 奖励集成 + 闭环迭代解决双重瓶颈
- **(b)** 真机 Franka 平台：两轮 World-VLA-Loop 迭代后，"Place Cup" 任务成功率 +36.7%，"Push Cube" 任务 +26.6%

### Figure 2: 现有世界模型在失败案例上的幻觉问题

![Figure 2](https://arxiv.org/html/2602.06508v2/x2.png)

**说明**: 现有视频世界模型对动作精度不敏感——透明叠加层显示真实夹爪轨迹存在微小偏差（位置偏移、姿态不对齐），导致真实环境中抓取失败，但视频模型仍然幻觉式地生成"成功抓取"的视频。这正是 [[SANS 数据集]] 要解决的核心问题。

### Figure 3: 完整流水线（4 阶段）

![Figure 3](https://arxiv.org/html/2602.06508v2/x3.png)

**说明**: World-VLA-Loop 的完整训练流程：
1. **Phase 1**: 通过遥操作 + 策略 rollout 构建 [[SANS 数据集]]
2. **Phase 2**: 在 SANS 上联合训练动作条件视频世界模型（视频 + 奖励双监督）
3. **Phase 3**: 在世界模型内 rollout VLA 策略，用预测奖励驱动 [[GRPO]] 优化
4. **Phase 4**: 部署新策略采集失败+成功数据，增强 SANS 进入下一轮迭代

### Figure 4: 沿训练步的成功率提升曲线

![Figure 4a — LIBERO-Object](https://arxiv.org/html/2602.06508v2/x4.png)
![Figure 4b — LIBERO-Goal](https://arxiv.org/html/2602.06508v2/x5.png)
![Figure 4c — LIBERO-Spatial](https://arxiv.org/html/2602.06508v2/x6.png)
![Figure 4d — Real-World](https://arxiv.org/html/2602.06508v2/x7.png)

**说明**: 4 个 benchmark 上的 RL 训练曲线均表现出**单调上升**态势，无明显震荡或崩溃。真机曲线（d）虽起点低（SFT≈13-27%）但收益最大（绝对提升 +13-23%），证明世界模型驱动的 RL 在真实场景同样有效。

### Figure 5: 世界模型生成 vs 真实执行对比

![Figure 5](https://arxiv.org/html/2602.06508v2/x8.png)

**说明**: 同一指令下展示 4 类视频对比：
- 世界模型对 SFT 策略动作的生成
- SFT 策略真实执行的视频
- 世界模型对 RL 后训练策略动作的生成
- RL 后训练策略真实执行的视频

二者在外观、轨迹、成败状态上高度对齐，验证世界模型作为虚拟环境的可信度。

### Figure 6: 真机实验装置

![Figure 6](https://arxiv.org/html/2602.06508v2/x9.png)

**说明**: 真机实验设置——[[Franka 研究臂]] + RealSense D435 第三视角固定相机。任务为 "Place Cup"（将杯子放到指定位置）与 "Push Cube"（推方块到目标区域）。

### Figure 7: 世界模型生成的失败轨迹

![Figure 7](https://arxiv.org/html/2602.06508v2/x10.png)

**说明**: 在 [[SANS 数据集]] 训练下，世界模型能够生成符合物理规律的**失败轨迹**：如夹爪未对准、夹爪过早闭合、物体滑落等，对应奖励头输出 $\hat{r}_t \approx 0$。这是与现有方法的核心差异。

### Figure 8: 世界模型生成的成功轨迹

![Figure 8](https://arxiv.org/html/2602.06508v2/x11.png)

**说明**: 世界模型在动作正确时也能生成清晰的成功轨迹，物体姿态、抓取过程、目标位置精确，对应奖励头输出 $\hat{r}_t \approx 1$。

### Figure 9: 未见动作序列上的泛化

![Figure 9](https://arxiv.org/html/2602.06508v2/x12.png)

**说明**: 测试世界模型对**训练分布外动作**的响应：
- **(a)** 盘子初始化在马克杯正前方
- **(b)** 夹爪右移 → 回到杯口上方 → 退到中性姿态
- **(c)** 夹爪先右移 → 然后成功 pick-and-place 杯子到盘子
- **(d)** 前后左右往复振荡

世界模型能合理跟随这些非常规动作序列，证明动作条件化的有效性而非过拟合训练轨迹。

### Table 1: 视频生成质量

| Scenario | [[SSIM]] ↑ | [[PSNR]] ↑ | [[LPIPS]] ↓ | [[MSE]] ↓ |
|----------|------|------|-------|-------|
| LIBERO | 0.90 | 26.57 | 0.031 | 0.0024 |
| Real-World | 0.91 | 29.61 | 0.059 | 0.0019 |
| **Average** | **0.91** | **28.09** | **0.045** | **0.0022** |

**说明**: 在 LIBERO 仿真与真机两类场景上的视频质量均达到[[结构相似性|高 SSIM]]（>0.90）与可接受的 [[PSNR]]（>26），证明世界模型能生成高保真的未来帧预测。

### Table 2: 结果对齐（视觉对齐 / 奖励对齐）

| Metric | LIBERO-Object T1 | LIBERO-Object T2 | LIBERO-Goal T1 | LIBERO-Goal T2 | LIBERO-Spatial T1 | LIBERO-Spatial T2 | Real Place Cup | Real Push Cube |
|--------|------|------|------|------|------|------|------|------|
| Visual Alignment | 92% | 90% | 94% | 78% | 86% | 94% | 90% | 84% |
| Reward Alignment | 88% | 90% | 90% | 76% | 88% | 94% | 94% | 78% |

**说明**: 每任务 50 个样本评估，**视觉对齐**衡量世界模型预测的视频与真实视频在关键动作节点是否一致，**奖励对齐**衡量预测奖励与真值是否匹配。两类对齐率普遍 >85%，奖励对齐与视觉对齐相关性高，验证联合监督的有效性。

### Table 3: VLA 策略成功率对比（核心结果）

| Model | LIBERO-Object T1 | LIBERO-Object T2 | LIBERO-Goal T1 | LIBERO-Goal T2 | LIBERO-Spatial T1 | LIBERO-Spatial T2 | Real Place Cup | Real Push Cube |
|-------|------|------|------|------|------|------|------|------|
| SFT Base | 73.9% | 73.9% | 91.9% | 86.1% | 83.9% | 87.9% | 13.3% | 26.7% |
| **RL Post-Training (Ours)** | **97.9%** | **91.9%** | **100%** | **96.2%** | **93.9%** | **94.0%** | **36.7%** | **40.0%** |
| Δ vs SFT | +24.0% | +18.0% | +8.1% | +10.1% | +10.0% | +6.1% | +23.4% | +13.3% |
| RL Post-Training (Oracle) | 98.7% | 98.7% | 98.8% | 98.8% | 98.2% | 98.2% | — | — |

**说明**:
- LIBERO 评估：500 次 rollout / 任务；真机：30 次物理 rollout
- **Oracle 上界**: 使用真实环境（而非世界模型）做 RL 的结果，本文方法在 LIBERO 上已接近上界（差距 <5%）
- 真机任务因 SFT 起点低，绝对收益最显著（+13-23%），证明世界模型驱动的 RL 在真实场景同样有效

### Table 4: 消融实验

| Metric | LIBERO-Object T1 | LIBERO-Object T2 | Real T1 |
|--------|------|------|------|
| **Visual Alignment** | | | |
| w/o near-success data | 60% | 66% | 54% |
| w/o reward prediction head | 68% | 70% | 80% |
| **Ours (Full)** | **92%** | **90%** | **90%** |
| **Reward Alignment** | | | |
| Qwen3-VL-8B-Instruct (外置 VLM) | 84% | 58% | 84% |
| **Ours (Full)** | **88%** | **90%** | **94%** |

**关键发现**:
1. **去掉 [[SANS 数据集|near-success 数据]]**: 视觉对齐骤降 ~30%（92% → 60%），说明失败样本对学习细粒度动力学至关重要
2. **去掉[[预测编码头|奖励预测头]]联合训练**: 视觉对齐降 ~20%（92% → 68%），证明奖励监督反过来也提升视频质量
3. **外置 [[Qwen3-VL]]-8B 当 reward critic**: 比集成奖励头低 4-32%，验证视觉-奖励潜表征共享的优势

### Table 5: 长时序生成质量衰减（Appendix B.1）

| Frame | [[PSNR]] | [[SSIM]] |
|-------|------|------|
| 200 | 23.59 | 0.743 |
| 250 | 22.21 | 0.673 |
| 300 | 20.24 | 0.628 |

**说明**: 自回归视频生成在 300 帧后出现明显质量衰减（[[PSNR]] <21、[[SSIM]] <0.65），这是本文方法应用于长时序任务的主要瓶颈。

### Table 6: VLM 奖励 vs 集成奖励头（Appendix B.2）

| Model | LIBERO-Object T1 | LIBERO-Object T2 |
|-------|------|------|
| **Ours (集成奖励头)** | **97.9%** | **91.9%** |
| Ours w/ VLM 奖励 | 93.9% | 88.0% |

**说明**: 即使将奖励头替换为 [[Qwen3-VL]]-8B VLM 评判，最终 RL 训练的策略成功率仍下降 4-4.7%，证明端到端集成的奖励信号优于外挂 VLM。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ManiSkill [[SANS 数据集]] | 35k 视频-动作对，23 任务 | 大规模预训练池 | 世界模型预训练 |
| [[LIBERO]] SANS | ∼50 succ + 50 fail / 任务 | 仿真任务微调 | 任务专属微调 |
| Real-World SANS | ∼50 succ + 50 fail / 任务 | 真机 Franka 平台 | 真机微调 |

### 实现细节

- **Backbone**: [[Cosmos-Policy|Cosmos-Predict 2]]（[[DiT]] 视频扩散）
- **VLA 策略**: SimpleVLA-RL 框架，[[GRPO]] 优化器
- **统一 chunk size**: 24 帧
- **动作表征**: 6-DoF [[末端执行器]]位姿 + 夹爪开合
- **真机硬件**: Franka research arm + RealSense D435（第三视角固定相机）
- **任务**: LIBERO（Object/Goal/Spatial 三套基准 × 2 任务）+ 真机 Place Cup / Push Cube
- **评估规模**: LIBERO 500 rollouts/task，真机 30 rollouts/task

### 可视化结果

- **失败案例正确建模**: Figure 7 显示世界模型能生成符合物理的失败（夹爪过早闭合、滑落等）
- **成功-失败鉴别**: 视觉对齐与奖励对齐 >85%，两者强相关
- **泛化性**: Figure 9 显示对训练外动作序列（如往复振荡）也能合理响应
- **2 轮迭代后大幅提升**: 真机 Place Cup 任务从 13.3% → 36.7%（+23.4 绝对，+176% 相对）

---

## 批判性思考

### 优点

1. **概念清晰、贡献正交**: SANS 数据 + 奖励集成 + 闭环迭代三个贡献彼此独立、消融可验证，论文叙事干净
2. **真机验证扎实**: 不只刷 LIBERO，真机 Franka 任务也展示显著提升，且公开评估规模（30 次物理 rollout）
3. **奖励集成设计优雅**: 共享 [[DiT]] 潜表征比外挂 [[Qwen3-VL]] VLM 更高效，潜在地形成视频-奖励互正则
4. **闭环范式具有可扩展性**: 一旦工程化，迭代次数与数据规模都可继续 scale up，是真实部署的合理路径

### 局限性

1. **长时序衰减**: 300 帧后 [[PSNR]]/[[SSIM]] 严重退化，限制应用于长 horizon 任务（如双手协作、多步装配）
2. **任务多样性有限**: 真机仅 2 个任务（Place Cup, Push Cube），LIBERO 也仅每基准 2 个任务，对范式的 stress test 不充分
3. **SANS 标注成本未量化**: "近似成功"轨迹的获取依赖人工挑选或策略 rollout 后的人工筛选，未给出完整成本估算
4. **没有公开代码/权重**: 复现门槛高，尤其 [[Cosmos-Policy|Cosmos-Predict 2]] 主干本身的可获取性存疑
5. **闭环迭代次数收益曲线缺失**: 论文展示了 2 轮迭代的结果，但未给出"第 3 轮、第 4 轮还能不能继续涨"的趋势分析
6. **奖励为二元**: 当前奖励仅 0/1 末态信号，对密集奖励、稀疏奖励的过渡如何处理无讨论

### 潜在改进方向

1. **稀疏-密集奖励混合**: 引入中间子目标奖励（如"抓住物体"、"到达放置区域上方"）应对长 horizon
2. **更强视频主干**: 用 [[CausVid]]、[[LTX-2]] 等更长 horizon 的视频模型替换 [[Cosmos-Policy|Cosmos-Predict 2]]
3. **自动 near-success 挖掘**: 用模型置信度或奖励边缘性自动筛选 near-success 轨迹，减少人工
4. **多模态奖励融合**: 结合本体感觉（力、触觉）作为多通道奖励监督
5. **任务族扩展**: 在 [[Metaworld]]、[[RoboCasa]]、Bridge-V2 等更大任务集上验证范式可扩展性

### 可复现性评估

- [ ] 代码开源（论文未提及）
- [ ] 预训练模型（论文未提及）
- [x] 训练细节完整（chunk size、backbone、数据规模均说明）
- [x] 数据集可获取（[[LIBERO]] 与 [[ManiSkill]] 公开；真机数据私有）
- [ ] 评估脚本公开（论文未提及）

---

## 关联笔记

### 基于

- [[Cosmos-Policy|Cosmos-Predict 2]]: 视频世界模型主干
- [[DiT]]: 扩散 Transformer 架构
- [[Flow Matching]]: 视频生成的训练目标
- [[GRPO]]: VLA 策略的 RL 算法
- SimpleVLA-RL: 策略后训练流水线（论文引用 baseline）

### 对比

- [[UniSim]]: 早期视频世界模型，仅成功轨迹训练
- [[IRASim]]: 视频世界模型用于策略训练
- [[Vid2World]]: 视频转世界模型范式
- [[WorldVLA]]: VLA 与世界模型联合训练
- [[WMPO]]: 基于视频 world model 的 policy optimization
- [[TGRPO]]: 时序 GRPO，另一种 VLA RL 后训练方案

### 方法相关

- [[SANS 数据集]]: 本文核心数据贡献
- [[状态感知视频世界模拟器]]: 本文核心模型贡献
- [[预测编码头]]: 奖励头设计
- [[EDM 噪声加权]]: 损失权重调制框架
- [[VLA]]: 任务范式

### 硬件/数据相关

- [[ManiSkill]]: 仿真基准 + 预训练数据来源
- [[LIBERO]]: 下游评估基准
- [[Franka 研究臂]]: 真机硬件平台

---

## 速查卡片

> [!summary] World-VLA-Loop
> - **核心**: SANS 数据 + 奖励集成 + 闭环迭代，三位一体让视频世界模型成为 VLA RL 后训练的可信虚拟环境
> - **方法**: [[Cosmos-Policy|Cosmos-Predict 2]] [[DiT]] 主干 + 轻量 [[MLP]] 奖励头 + [[GRPO]] 优化器
> - **结果**: 真机 Franka 两任务两轮迭代后 +36.7% / +26.6%，LIBERO 三基准全部接近 Oracle 上界
> - **关键消融**: 去 near-success 数据 → 视觉对齐 -30%；外置 [[Qwen3-VL]] reward → -4~32%
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-05-26*
