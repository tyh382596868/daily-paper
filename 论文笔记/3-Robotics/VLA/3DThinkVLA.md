---
title: "3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training"
method_name: "3DThinkVLA"
authors: [Jiaxin Shi, Xidong Zhang, Fucai Zhu, Zhe Li, Siyu Zhu, Weihao Yuan]
year: 2026
venue: arXiv
tags: [vision-language-action, 3d-spatial-reasoning, knowledge-distillation, co-training, latent-alignment, embodied-ai]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.04436v1
created: 2026-06-07
---

# 论文笔记：3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University, Harbin Institute of Technology, NTU, Fudan University, Nanjing University, Daimon Robotics, Great Bay University |
| 日期 | June 2026 |
| 项目主页 | — (代码承诺公开但尚未提供仓库链接) |
| 对比基线 | [[OpenVLA-OFT]], [[SpatialVLA]], [[CoT-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04436) / [HTML](https://arxiv.org/html/2606.04436v1) |

---

## 一句话总结

> 通过 3D 几何感知 + 3D 推理蒸馏的协同训练，把 3D 空间智能"压"进 [[VLA]] 的潜空间，部署时只用 2D 图像即可隐式 3D 推理。

---

## 核心贡献

1. **解耦 3D 几何与 3D 推理**: 把 "几何感知" 和 "空间推理" 视为两种可解耦的能力，分别在视觉中间层（低层）和动作前置 token（高层）注入。
2. **诊断并解决 Prompt-Induced Reasoning Gap**: 首次明确指出 vanilla 3D 协同训练存在"动作 prompt 一来就关闭空间推理"的现象，提出 **Reasoning Anchor Token + Online Latent Distillation** 在不生成 CoT 文本的前提下迁移高层空间思维。
3. **部署轻量化**: 推理阶段丢弃 3D 基础模型与 teacher 分支，只保留两个 adapter，纯 2D 输入即可实现隐式 3D 推理，在 [[LIBERO]] / LIBERO-PLUS / [[SimplerEnv]] / 真机均刷到 SOTA。

---

## 问题背景

### 要解决的问题

如何让 [[VLA]] 模型在**不引入额外 3D 传感器、不依赖外部 3D 基础模型、不显式生成 [[CoT|思维链]]**的前提下，依然具备 3D 空间智能（深度推断、相对位置推断、遮挡推断）？

### 现有方法的局限

- **显式 3D 输入流派**（SpatialVLA、PTv3、[[DP3]]）: 需要 point cloud / depth 等 3D 模态，依赖额外传感器与编码器，部署成本高。
- **显式 CoT 推理流派**（[[CoT-VLA]]）: 推理时需要先生成空间推理文本再生成动作，**推理延迟翻倍**，且文本质量与下游动作脱节。
- **Vanilla 3D 协同训练**（直接混入 3D QA 数据训 VLM 部分）: 作者发现存在 **[[Prompt-Induced Reasoning Gap]]** —— 当 prompt 切换到"action prompt"时，模型会走 **[[Action Shortcut|动作捷径]]**，关闭刚学到的空间推理能力（见 Figure 1(b)、Figure 5(b)）。

### 本文的动机

> 几何感知是低层视觉先验，空间推理是高层语言-视觉联合推理。**不应该用同一个 prompt 同一个层级注入**，应该分层注入并用 **[[Latent Distillation|隐式蒸馏]]** 强制 action prompt 路径保留 reasoning prompt 路径的空间思维。

---

## 方法详解

### 模型架构

<!-- 内联概念链接所有技术术语 -->

3DThinkVLA 采用 **VLM Backbone + Action Head + 两个轻量 Adapter** 架构，主体基于 [[Qwen3-VL]]-2B + [[OpenVLA-OFT]] 风格的 action head：

- **输入**: 2D RGB 图像 $I_t$ + 任务指令 $L_{task}$ + Action 指令 $L_{action}$
- **Backbone**: Qwen3-VL-2B（带 ViT 视觉编码器）
- **核心模块**:
  - [[Geometry Adapter|几何 Adapter]]（[[MLP]] + [[LayerNorm]]）插在 ViT 第 18 层，把中间视觉 token 对齐到 [[VGGT]] 的 3D patch token
  - [[Reasoning Adapter|推理 Adapter]]（MLP）把 student 分支的 anchor token 投影到 teacher 的 reasoning 潜空间
  - [[Reasoning Anchor Token|推理锚点 token]] $\tau_R$ 作为可学习的"空间思维容器"
- **三步前向**: 几何对齐 → 推理蒸馏 → 空间增强的动作融合（element-wise add）
- **输出**: [[Action Chunking|动作块]] $\hat{A}_t = a_{t:t+k}$
- **总参数**: ~2B（同 Qwen3-VL-2B 同量级，adapter 仅 ~MB 级）

### 核心模块

#### 模块1: Latent 3D Geometry Perception（低层几何注入）

**设计动机**: 让视觉中间层带上**低层几何先验**（深度、表面法向、相对距离），但**不改 VLM 主干结构**，避免破坏预训练知识。

**具体实现**:
- 从 ViT 第 18 层取出 patch-level visual token $\mathcal{F}^{Vis}$
- 通过 [[Geometry Adapter|几何 Adapter]]（MLP + LayerNorm）映射到 $\mathcal{F}^{Geo}$
- 用一个**冻结的** [[VGGT]] （3D 基础模型）作为 teacher，提取 3D-aware patch token $\mathcal{F}^{3D}$
- 用 patch-level [[Cosine Similarity|余弦相似度]] 对齐：$\mathcal{L}_{geo} = 1 - \mathcal{S}(\mathcal{F}^{3D}, \mathcal{F}^{Geo})$

**关键设计**: VGGT 仅在训练时使用，推理时丢弃 → 几何先验已"压"进 ViT 与 Geometry Adapter。

#### 模块2: Online 3D Reasoning Distillation（高层推理注入）

**设计动机**: 解决 [[Prompt-Induced Reasoning Gap]] —— 当 prompt 是 "action prompt" 时，模型走 [[Action Shortcut|动作捷径]]，丢弃空间推理。

**具体实现**:
- 引入共享的 **[[Reasoning Anchor Token|reasoning anchor token]]** $\tau_R$，插在 $L_{task}$ 之后
- **Teacher 分支**（带 explicit 3D reasoning prompt $L^{teacher}$）：
  $$H^R_{teacher} = sg(f_\theta(I_t, L_{task}, L^{teacher}, \tau_R))$$
  其中 $sg(\cdot)$ 是 stop-gradient
- **Student 分支**（带 action prompt $L_{action}$、action anchor $\tau_A$）：
  $$\hat{H}^R_{student} = f_\theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)$$
- 用 [[Reasoning Adapter|reasoning adapter]] $\mathcal{R}$ 把 student anchor 投到 teacher 潜空间，做 cosine 蒸馏：
  $$\mathcal{L}_{reasoning} = 1 - \mathcal{S}(H^R_{teacher}, \mathcal{R}(\hat{H}^R_{student}))$$

**关键设计**: VLM 协同训练阶段 reasoning anchor 作为**第一个输出 token**，强制自回归条件依赖空间先验 → 推理时无需生成 CoT 文本。

#### 模块3: Spatially Augmented Action Integration（双路融合）

**设计动机**: 几何先验与推理先验来自不同抽象层，需要在动作端**对齐到 action latent 空间再融合**，防止 action head 走捷径。

**具体实现**:
- 把 $\mathcal{F}^{Geo}$ 与 $\mathcal{R}(\hat{H}^R_{student})$ 各自投影到 action token 维度 → $H^A_{geo}$、$H^A_{reasoning}$
- 用 **element-wise add** 融入 action query token $H_A$：
  $$\hat{A}_t = \mathrm{ActionHead}(H_A + H^A_{geo} + H^A_{reasoning})$$
- 训练时对空间分支做随机 dropout 正则，防止 action head 过度依赖某条路径

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

### 公式1: [[VLA|动作预测]] 标准定义

$$
\hat{A}_t = \pi_\Theta(I_t, L, \tau_A)
$$

**含义**: 标准 VLA 把动作建模为给定图像 $I_t$、指令 $L$ 和 action anchor $\tau_A$ 的策略输出。本文以此为起点重构。

**符号说明**:
- $\pi_\Theta$: 策略网络，参数 $\Theta$
- $I_t$: 第 $t$ 步 2D RGB 图像
- $L$: 自然语言任务指令
- $\tau_A$: action anchor token
- $\hat{A}_t$: 预测动作块（chunk）

### 公式2: [[Cosine Similarity|几何对齐]] 损失

$$
\mathcal{L}_{geo} = 1 - \mathcal{S}(\mathcal{F}^{3D}, \mathcal{F}^{Geo})
$$

**含义**: 把 VLM 中间层视觉特征拉向冻结 VGGT 的 3D 特征，patch 级别 cosine 距离。

**符号说明**:
- $\mathcal{S}(\cdot, \cdot)$: patch-wise [[Cosine Similarity|余弦相似度]] 平均
- $\mathcal{F}^{3D}$: VGGT 输出的 3D-aware patch token（teacher）
- $\mathcal{F}^{Geo}$: 经 [[Geometry Adapter]] 映射后的 VLM 中间层 token

### 公式3: 重构后的 [[VLA|动作预测]]（带 anchor）

$$
\hat{A}_t = \pi_\Theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)
$$

**含义**: 把指令拆为 $L_{task}$ 与 $L_{action}$，在两者之间插入 reasoning anchor $\tau_R$，让自回归模型在生成 action anchor 之前必先"经过"空间思维。

**符号说明**:
- $\tau_R$: [[Reasoning Anchor Token|reasoning anchor]]，可学习
- $L_{task}$: 任务描述（"pick up the red bowl"）
- $L_{action}$: 动作指令模板（"predict the next action chunk"）

### 公式4: Teacher 分支 reasoning anchor 提取

$$
H^R_{teacher} = sg(f_\theta(I_t, L_{task}, L^{teacher}, \tau_R))
$$

**含义**: teacher 用 explicit 3D 推理 prompt $L^{teacher}$（如"What is the depth of objects relative to camera?"）激发 VLM 的空间推理表征，stop-gradient 后作为蒸馏目标。

**符号说明**:
- $f_\theta$: 共享 VLM backbone
- $sg(\cdot)$: stop-gradient 算子
- $L^{teacher}$: 显式 3D reasoning prompt

### 公式5: Student 分支 reasoning anchor 提取

$$
\hat{H}^R_{student} = f_\theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)
$$

**含义**: student 用 action prompt 路径前向，得到的 anchor 隐藏态作为蒸馏源。注意 student 与 teacher **共享同一组 VLM 参数**（online distillation）。

### 公式6: [[Latent Distillation|隐式推理蒸馏]] 损失

$$
\mathcal{L}_{reasoning} = 1 - \mathcal{S}(H^R_{teacher}, \mathcal{R}(\hat{H}^R_{student}))
$$

**含义**: 用 cosine 距离对齐 teacher 与 student 的 reasoning anchor 表征，关键是中间用 [[Reasoning Adapter]] $\mathcal{R}$ 缓解 action / reasoning prompt 路径的分布差异。

**符号说明**:
- $\mathcal{R}(\cdot)$: [[Reasoning Adapter]]（MLP），可训练
- $H^R_{teacher}$: teacher 隐藏态（无梯度）
- $\hat{H}^R_{student}$: student 隐藏态（有梯度）

### 公式7: 空间增强的动作融合

$$
\hat{A}_t = \mathrm{ActionHead}(H_A + H^A_{geo} + H^A_{reasoning})
$$

**含义**: 几何与推理两路空间先验经投影后与 action query token 做**元素加**，再喂给 action head。Element-wise add 在消融中胜过 cross-attention 和 gate fusion（见 Table 6）。

**符号说明**:
- $H_A$: action query token 隐藏态
- $H^A_{geo}$: 几何先验在 action 空间的投影
- $H^A_{reasoning}$: 推理先验在 action 空间的投影

### 公式8: VLA-step 总目标

$$
\mathcal{L}_{vla} = \mathcal{L}_{action} + \lambda_a \mathcal{L}_{geo} + \lambda_d \mathcal{L}_{reasoning}
$$

**含义**: VLA 数据迭代时同时优化动作回归 + 几何对齐 + 推理蒸馏三项。

**符号说明**:
- $\mathcal{L}_{action}$: 动作 [[L1 损失|L1]] / 回归损失
- $\lambda_a, \lambda_d$: 权重，论文设为 $0.5$
- 此 step **仅训练 adapter 与 action head**，VLM backbone 冻结或低 lr

### 公式9: [[Co-training|协同训练]] VLM 损失

$$
\mathcal{L}_{vlm} = \lambda_{3D} \mathcal{L}_{CE}
$$

**含义**: 3D reasoning 数据迭代时，用 explicit 3D QA（"What's behind the red bowl?"等）做标准 next-token CE 训练 VLM，要求 reasoning anchor 作为**第一个输出 token**。

**符号说明**:
- $\mathcal{L}_{CE}$: cross-entropy 损失，对答案 token 计算
- $\lambda_{3D}$: 权重

### 公式10: 总训练目标

$$
\mathcal{L}_{total} = \mathcal{L}_{vla} + \mathcal{L}_{vlm}
$$

**含义**: 两类数据按 VLA:VLM = 1.0:0.1 的比例混合训练，让 VLM 同时擅长"空间推理回答"和"无 CoT 直接出动作"。

---

## 关键图表

### Figure 1: Overview / 框架总览与 Reasoning Gap 现象

![Figure 1](https://arxiv.org/html/2606.04436v1/x1.png)

**说明**:
- (a) 框架: VLM backbone 在 VLA 数据 + 真实 3D reasoning 数据上协同训练；
- (b) **关键观察 — [[Prompt-Induced Reasoning Gap]]**: vanilla co-training 模型在 action prompt 下会退化到 [[Action Shortcut|动作捷径]]，关闭空间感知；
- (c) 本文用 [[Reasoning Adapter]] + online [[Latent Distillation]] 把空间思维注入 action prediction，**无需显式文本生成**；
- (d)(e) 在 [[LIBERO]] / LIBERO-PLUS 上的定性结果。

### Figure 2: 3D-Thinking-Guided Framework / 完整架构

![Figure 2](https://arxiv.org/html/2606.04436v1/x2.png)

**说明**: 框架的两个数据流：
- (a) VLM backbone 在 VLA 与 3D 数据上协同训练，通过显式 3D reasoning 任务**内化**空间智能；
- (b) [[Geometry Adapter]] 把 ViT 中间层视觉特征通过 patch-level [[Latent Alignment|潜空间对齐]] 拉向 [[VGGT]]，获取低层几何先验。
- 推理时仅保留 adapter，丢弃 VGGT 与 teacher 分支。

### Figure 3: Qualitative Results on LIBERO-Plus

![Figure 3](https://arxiv.org/html/2606.04436v1/fig/success1.png)

**说明**:
- Task 1: "Put the black bowl in the **bottom** drawer of the cabinet and close it."
- Task 2: "Put the black bowl in the **top** drawer of the cabinet and close it."
- 关键展示: 3DThinkVLA 能正确区分 bottom / top（依赖深度与垂直空间推理），而基线（OpenVLA-OFT）在这种"语言细微差别 + 空间推理"组合下失败。

### Figure 4: Real-world Robot Setup

![Figure 4 - Robot](https://arxiv.org/html/2606.04436v1/fig/robotsetup_a.png)
![Figure 4 - Tasks](https://arxiv.org/html/2606.04436v1/fig/robotsetup_b.png)

**说明**:
- (a) **Realman 机器人**: 7-DoF 机械臂 + 1-DoF 夹爪 + 顶部相机 + 腕部相机；
- (b) 三个真实任务：
  - **Task 1 — Height Variation**: 把物体放到不同高度的架子（考验**深度感知**）
  - **Task 2 — Transparent Container**: 操作透明容器（考验**几何理解**）
  - **Task 3 — Spatial Positioning**: 精确空间定位（考验**空间推理**）

### Figure 5: Action Loss Curve & Projection-Space Similarity Analysis

![Figure 5](https://arxiv.org/html/2606.04436v1/x3.png)

**说明**:
- (a) **训练动作损失曲线**: 蓝线（本文 online 3D reasoning distillation）显著低于红线（仅 3D co-training），证明蒸馏帮助动作监督收敛；
- (b) **Projection-Space Similarity Analysis** — 量化 action prompt 与 reasoning prompt 路径在 anchor 潜空间的相似度。蓝线（带蒸馏）相似度持续高于 0.8，红线（仅协同训练）则在训练中后期下降到 0.5 以下 → **直接定量证明 [[Prompt-Induced Reasoning Gap]] 的存在与消除**。

### Table 1: [[LIBERO]] 基准

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| [[OpenVLA]] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[CoT-VLA]] | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| [[OpenVLA-OFT]] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| [[SpatialVLA]] | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| SpatialForcing | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| VITA | 95.9 | 98.9 | 95.1 | 96.8 | 96.7 |
| **3DThinkVLA (Ours)** | **100.0** | **100.0** | **98.8** | 95.8 | **98.7** |

**说明**: 在 LIBERO 四子集上刷到 SOTA，Spatial / Object 两项满分。Long 略低于 VITA 与 GeoVLA，但平均最高。

### Table 2: LIBERO-Plus（鲁棒性扰动）

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|--------|--------|-------|----------|-------|------------|-------|--------|-----|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.9 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| ABot-M0 | 60.4 | **67.9** | **86.4** | 96.2 | 91.6 | **86.4** | **82.6** | 80.5 |
| Qwen3-VL-OFT | 47.0 | 60.1 | **87.0** | 96.3 | **95.3** | 73.1 | 79.2 | 75.0 |
| **3DThinkVLA (Ours)** | **73.8** | 64.5 | 78.0 | **98.4** | 94.8 | 84.7 | 81.5 | **81.0** |

**说明**: LIBERO-Plus 加入七种扰动（相机视角、机器人差异、语言改写、光照、背景、噪声、布局）。3DThinkVLA 平均最高，**Camera 扰动**（视角变化最考验 3D 理解）提升 13 pt（60.4→73.8），证明 3D 先验真的迁移了。

### Table 3: [[SimplerEnv]] (WidowX)

| Method | Put Carrot on Plate | Put Eggplant in Basket | Put Spoon on Towel | Stack Block | Avg |
|--------|--------|---------|---------|---------|------|
| Octo | 8.3 | 43.1 | 12.5 | 0.0 | 16.0 |
| OpenVLA | 0.0 | 4.1 | 0.0 | 0.0 | 1.0 |
| RoboVLM | 20.8 | 79.2 | 45.8 | 4.2 | 37.5 |
| [[SpatialVLA]] | 25.0 | **100.0** | 16.7 | 29.2 | 42.7 |
| Open π0 | 61.3 | 89.6 | 73.7 | 15.8 | 60.0 |
| QDepth-VLA | 57.5 | 95.0 | 82.0 | 39.6 | 68.5 |
| FALCON | 41.7 | **100.0** | 62.5 | 20.8 | 56.3 |
| UniVLA | **83.3** | 66.7 | 33.3 | **95.8** | 69.8 |
| VITA | 68.8 | 95.6 | 84.2 | 37.5 | 71.5 |
| **3DThinkVLA (Ours)** | 75.0 | 95.8 | **87.5** | 33.3 | **72.9** |

**说明**: SimplerEnv 上平均 72.9% 超 VITA 1.4pt，Spoon-on-Towel 子任务（精细放置）最高。Stack Block 任务略弱于 UniVLA，作者归因于堆叠任务依赖时序而非空间推理。

### Table 4: 消融 — 协同训练数据 + Adapter 设计

| ID | Co-training | Geometry Adapter | Reasoning Adapter | Reasoning Anchor | Spatial | Object | Goal | Long | Avg |
|----|------------|-----------------|------------------|-----------------|---------|--------|------|------|------|
| R1 | – | ✗ | ✗ | ✗ | 93.6 | 99.6 | 97.4 | 92.6 | 95.8 |
| R2 | 2D Data | ✗ | ✗ | ✗ | 96.8 | 99.8 | 98.0 | 95.0 | 97.4 |
| R3 | 3D Data | ✗ | ✗ | ✗ | 99.8 | 100.0 | 98.8 | 93.0 | 97.9 |
| R4 | 3D Data | ✓ | ✗ | ✗ | 99.0 | 100.0 | 98.2 | 95.8 | 98.3 |
| R5 | 3D Data | ✓ | ✓ | ✗ | 99.6 | 100.0 | 98.8 | 95.8 | 98.6 |
| R6 | 3D Data | ✓ | ✗ | ✓ | 99.2 | 100.0 | 99.4 | 94.4 | 98.3 |
| R7 | 3D Data | ✓ | ✓ | ✓ | 100.0 | 100.0 | 98.8 | 95.8 | **98.7** |

**关键发现**:
- R1→R3: 3D 协同训练已带来 2.1 pt 提升（**Spatial 子集大涨 6.2 pt**）
- R3→R4: 加 Geometry Adapter 再涨 0.4 pt（Long 子集 +2.8 pt）
- R4→R7: 加 Reasoning Anchor + Distillation 再涨 0.4 pt，**Spatial 满分**
- R5 vs R6: 单独加 Reasoning Adapter（无 anchor）或单独加 Anchor（无 Adapter）效果接近，**两者协同才最优**

### Table 5: 真机任务

| Method | Task 1 (Height Variation) | Task 2 (Transparent Container) | Task 3 (Spatial Position) |
|--------|--------------------------|-------------------------------|-------------------------|
| π0 | 63.3% | 82.0% | 51.3% |
| [[OpenVLA-OFT]] | 28.0% | 57.3% | 30.7% |
| **3DThinkVLA (Ours)** | **88.0%** | **93.3%** | **61.3%** |

**说明**: 真机三个**专门考验空间理解**的任务全部刷新 SOTA，Height 任务相对 π0 +24.7 pt 是最关键的证据。

### Table 6: 消融 — Reasoning 信息融合方式

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|------|
| [[Cross-Attention]] | 99.4 | 99.8 | 99.0 | 93.6 | 98.0 |
| **Add Fusion (Ours)** | **100.0** | **100.0** | 95.8 | 98.6 | **98.7** |
| Gate Fusion | 99.6 | 99.8 | 99.2 | 95.0 | 98.4 |

**说明**: 简单的 element-wise add 优于 cross-attention 与 gate fusion，作者解释为：低秩 add 不破坏 action query 的原始分布，且不引入额外参数瓶颈。

### Table 7: 消融 — 蒸馏信号类型

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|------|
| Baseline | 99.0 | 100.0 | 98.2 | 95.8 | 98.3 |
| **Reasoning Anchor (Ours)** | **100.0** | **100.0** | 95.8 | 98.6 | **98.7** |
| Privileged Info | 99.4 | 99.8 | 98.6 | 94.4 | 98.1 |

**说明**: 用 anchor 隐藏态做蒸馏优于"特权信息"（直接喂 3D 标签给 student）。anchor 蒸馏是**结构化、对齐输入分布**的；特权信息会引发训练-推理分布偏移。

### Table 8: 消融 — 3D 信息注入方式

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|------|
| Baseline | 93.6 | 99.6 | 97.4 | 92.6 | 95.8 |
| PTv3 | 98.6 | 99.6 | 99.2 | 91.6 | 97.3 |
| [[DP3]] Encoder | 93.8 | 100.0 | 98.0 | 94.2 | 96.5 |
| **Geometry Adapter (Ours)** | **98.2** | **100.0** | 97.4 | **94.8** | **97.6** |

**说明**: 隐式潜空间对齐胜过 explicit 3D encoder（PTv3 / DP3）。即：把 3D 知识"压"进 ViT 中间层比把 3D 模态显式喂进来更鲁棒。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] (Spatial/Object/Goal/Long) | 4×500 demos | 标准 VLA 仿真基准 | 训练+测试 |
| **LIBERO-Plus** | 7 类扰动 | LIBERO 的鲁棒性变体（相机/光照/语言/背景/布局） | 测试 |
| [[SimplerEnv]] (WidowX) | 4 个任务 | 真实图像 sim 基准 | 测试 |
| **SpatialReasoning-CoT** | ~24K 样本 | 来自 Spatial Reasoner，用 [[SAM2]] + [[Depth Anything V2]] + 位姿估计生成 | 3D co-training |
| 一般 image-text | ~24K 样本 | 与 SpatialReasoning-CoT 1:1 混合 | 防止 VLM 灾难性遗忘 |
| 真机 Realman | 3 个任务 | 7-DoF + 双相机 | 真机评估 |

### 实现细节

- **Backbone**: [[Qwen3-VL]]-2B + [[OpenVLA-OFT]] 风格 action head
- **优化器**: AdamW，混合精度 (bfloat16)
- **学习率**: backbone 2.5×10⁻⁵，interface 层 1.0×10⁻⁵，action head 2.0×10⁻⁴
- **Loss 权重**: VLA:VLM = 1.0:0.1，alignment / distillation 各 0.5
- **硬件**: 8× NVIDIA A100
- **训练开销**: 约为 vanilla VLA 的 1.5×（额外 teacher 前向 + VGGT 前向）
- **推理开销**: 与 vanilla VLA **几乎相同**（丢弃 VGGT 与 teacher）

### 可视化结果

- LIBERO-Plus 上 height-variation 子任务: 3DThinkVLA 能正确区分 "top drawer" / "bottom drawer"，OpenVLA-OFT 多数失败（Figure 3）
- 真机 transparent container 任务: 3DThinkVLA 能稳定抓取透明杯子，π0 偶发误判位置（Task 2 数据 93.3% vs 82.0%）
- Projection-Space Similarity Analysis（Figure 5b）: 量化证据表明蒸馏后 action 路径与 reasoning 路径的 anchor 表征相似度持续高于 0.8

---

## 批判性思考

### 优点

1. **诊断精准**: "Prompt-Induced Reasoning Gap" 是真实存在且**可量化**的现象（Figure 5b 的相似度曲线），不是讲故事。
2. **解耦优雅**: 几何（低层 ViT 中间层）和推理（高层 anchor token）分层注入的思路很自然，且 Table 8 证明隐式注入优于显式 3D encoder。
3. **部署友好**: 推理时丢掉 VGGT 与 teacher，**真正零额外开销**地获得 3D 能力，对真机部署友好。
4. **真机三任务**全部针对空间推理设计（高度变化、透明容器、精确定位），评估**针对性强**。

### 局限性

1. **3D reasoning 数据依赖伪标签**: SpatialReasoning-CoT 用 [[SAM2]] + [[Depth Anything V2]] 自动生成，**标签噪声**会上限化 teacher 的能力。论文只用了 24K 样本，scaling 行为未探索。
2. **Long-horizon 不是 SOTA**: LIBERO-Long 95.8% 略低于 GeoVLA / VITA，可能是 3D 蒸馏对**时序推理**没有直接帮助，长链任务还是受 action chunking 的 horizon 限制。
3. **Teacher prompt 设计不透明**: $L^{teacher}$ 的具体内容（论文里只举了 "What's the depth of the red bowl?" 这种例子）没有公开列出全部模板，复现性受影响。
4. **未对比 explicit CoT 蒸馏 baseline**: 一个自然的对照是"让 student 也生成 reasoning text 再蒸馏"，可惜没有。
5. **代码尚未公开**: 论文只承诺公开，目前没有仓库链接，复现风险中等。

### 潜在改进方向

1. **Reasoning Anchor 多 token**: 当前只有一个 $\tau_R$，改为多个不同语义的 anchor（深度 / 朝向 / 遮挡）可能更细粒度。
2. **VLA + VLM 数据比例自适应**: 现在固定 1.0:0.1，可以引入 dynamic weighting（按 reasoning 分歧度动态调）。
3. **结合 explicit depth supervision**: Geometry Adapter 当前只对齐 VGGT 特征，可以加一路 depth-map 直接监督进一步固化几何。
4. **拓展到双手 / 全身**: 真机只测了单臂 7-DoF，bimanual / mobile manipulation 上的空间推理收益值得验证。

### 可复现性评估
- [ ] 代码开源（承诺中）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（学习率、batch size、loss 权重均给出）
- [x] 数据集可获取（LIBERO / SimplerEnv 公开，SpatialReasoning-CoT 来自公开工具链）

---

## 关联笔记

### 基于
- [[Qwen3-VL]]: VLM backbone
- [[OpenVLA-OFT]]: action head 风格
- [[VGGT]]: 提供低层 3D 几何先验的 teacher

### 对比
- [[SpatialVLA]]: 显式 3D 输入流派，3DThinkVLA 主张隐式注入更优
- [[CoT-VLA]]: 显式 CoT 文本流派，3DThinkVLA 主张潜空间蒸馏更高效
- [[OpenVLA-OFT]]: 同 action head 设计的强 baseline
- [[DP3]]: 显式 3D encoder baseline，被 Table 8 击败

### 方法相关
- [[Co-training]]: VLA + VLM 协同训练范式
- [[Latent Distillation]]: 在隐藏态而非输出 logits 上做蒸馏
- [[Knowledge Distillation]]: 蒸馏总框架
- [[Geometry Adapter]]: 本文新概念
- [[Reasoning Adapter]]: 本文新概念
- [[Reasoning Anchor Token]]: 本文新概念
- [[Prompt-Induced Reasoning Gap]]: 本文新概念
- [[Action Shortcut]]: 本文新概念
- [[Embodied Spatial Intelligence]]: 上位概念
- [[Cosine Similarity]]: 用作对齐与蒸馏的相似度度量

### 硬件 / 数据相关
- [[LIBERO]]: 仿真基准
- [[SimplerEnv]]: sim-to-real 中间基准
- [[SAM2]]: 数据流水线中的分割
- [[Depth Anything V2]]: 数据流水线中的深度估计

---

## 速查卡片

> [!summary] 3DThinkVLA (arXiv 2606.04436, 2026.06)
> - **核心**: 把 3D 几何（低层）+ 3D 推理（高层）分别注入 [[VLA]] 潜空间，部署时只留 adapter
> - **方法**: Geometry Adapter 对齐 [[VGGT]] + Reasoning Anchor Token + online latent distillation
> - **关键洞见**: 诊断出 Prompt-Induced Reasoning Gap，用 anchor token 蒸馏修复
> - **结果**: LIBERO 98.7（SOTA）/ LIBERO-Plus 81.0（SOTA）/ SimplerEnv 72.9（SOTA）/ 真机三任务全部 SOTA
> - **代码**: 承诺公开，暂未发布

---

*笔记创建时间: 2026-06-07*
