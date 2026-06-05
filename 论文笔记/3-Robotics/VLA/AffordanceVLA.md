---
title: "AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Perception"
method_name: "AffordanceVLA"
authors: [Qize Yu, Jiadi You, Yuran Wang, Jiaqi Liang, Bowen Ping, Yang Tian, Yue Chen, Minghong Cai, Zeying Gong, Ruihai Wu, Yinchuan Li, Junwei Liang, Yingcong Chen]
year: 2026
venue: arXiv
tags: [vla, affordance, mixture-of-transformer, robot-manipulation, perception-action, embodied-ai, visual-grounding]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.06155
created: 2026-06-05
---

# 论文笔记：AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Perception

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 北京大学（PKU）；香港科技大学广州（HKUST-GZ）；香港中文大学（CUHK） |
| 日期 | June 2026 |
| 项目主页 | https://skywalker-yqz.github.io/AffordanceVLA/ |
| 对比基线 | [[Pi05\|π₀]] / [[OpenVLA]] / [[SpatialVLA]] / [[CoT-VLA]] / [[GR-1]] / [[UniVLA]] / [[Octo]] / Seer-Large / F1-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06155) / [Code](https://github.com/Skywalker-yqz/AffordanceVLA/) / [Project](https://skywalker-yqz.github.io/AffordanceVLA/) |

---

## 一句话总结

> 用 **Which–Where–How** 三层 [[Affordance Forecasting|结构化可供性预测]] 作为感知到动作的"任务导向中间表征"，并以 [[MoT|Mixture-of-Transformer]] 三专家 + [[UAA 注意力机制|UAA 渐进注意力]] 实现感知–规划–动作的解耦联训。

---

## 核心贡献

1. **结构化可供性预测作为中间表征**: 不再让 [[VLA]] 直接从像素映射到动作，而是显式预测 **Which2Act**（目标对象在哪）、**Where2Act**（2D 可交互区域）、**How2Act**（3D 形状 + 10-DoF 位姿布局）三层 [[Affordance|可供性]]，作为感知到动作的精确鲁棒桥梁。
2. **Mixture-of-Transformer 三专家架构**: 设计 [[Understanding Expert]]（语义对齐）、[[Affordance Generation Expert]]（结构化可供性预测）、[[Action Expert]]（低层动作生成）三个独立的 [[MoT]] 专家，分别处理感知、规划、控制三类异质任务，避免参数共享时的相互干扰。
3. **UAA 渐进注意力机制**: 提出 **Understanding–Affordance–Action (UAA)** 渐进注意力，跨专家用 **因果掩码** 保证可供性不被动作信息污染，专家内用 **双向注意力** 充分融合，防止"未来动作"反向泄漏到"预测阶段"。
4. **三阶段课程式训练**: Stage I 在 [[AGD20K]] / [[RefSpatial]] / [[PRISM]] 上预训练可供性专家；Stage II 在大规模合成机器人数据（[[InternData-A1]]）上联合训练动作和可供性（>100K 自动标注）；Stage III 下游微调时退火 [[Affordance Annealing|可供性权重]] 到 0.15，让控制精度优先。
5. **SOTA + 显著的数据效率**: [[LIBERO]] 平均 95.8%，[[CALVIN]] ABC→D 链长 4.33，真实机器人复杂任务平均 82.9%（[[Pi05\|π₀]] 仅 44.8%）；仅用 40% 训练数据即超过 [[Pi05\|π₀]] 满数据性能。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型直接从视觉-语言映射到机器人动作时，**感知-动作映射是隐式的**：模型必须同时学会"看懂场景"和"控制电机"，二者高度耦合，导致：
- 对未见过的物体、新场景、新指令泛化差；
- 长程任务里的小感知误差被动作累积放大；
- 缺少可解释的中间状态，难以调试和对齐人类意图。

### 现有方法的局限

- [[OpenVLA]] / [[Octo]] 等端到端 VLA 把感知和控制混在一个 transformer 里，动作头干扰了视觉表征学习。
- [[CoT-VLA]] / [[GR-1]] 等用未来帧预测或自然语言 CoT 作为中间表征，但视频帧生成代价高、与动作的几何关系弱；自然语言中间步骤缺乏空间精度。
- 基于 affordance map 的方法（如 [[Where2Act 系列]]）通常只有 2D 热力图，缺 3D 几何，难以指导 6-DoF/10-DoF 操作。

### 本文的动机

如果中间表征同时包含 **What（哪个物体）+ Where（哪个区域）+ How（什么 3D 形状/位姿）**，就能给动作专家提供**任务对齐、空间精确、几何完整**的提示。把这套结构化 [[Affordance|可供性]] 当作 explicit 中间表征，再用 MoT 让专家分工，就能解决感知-动作耦合问题。

---

## 方法详解

### 模型架构

<!-- 用结构化列表而非 ASCII 图描述 -->

AffordanceVLA 是一个三专家 [[MoT|Mixture-of-Transformer]] 架构：

- **输入**: 语言指令 $l$ + 多视角图像 $I_t$ + 机器人本体状态 $s_t$
- **Backbone**: 基于预训练 [[VLM]] 的视觉编码器（实验中 Stage I 冻结），文本以 token 形式注入
- **三个专家**（独立参数，token 序列共享但每个 layer 内部各算各的 FFN）：
  - **[[Understanding Expert]]** 输出语义对齐表征 $h^{und}_t$
  - **[[Affordance Generation Expert]]** 输出结构化可供性表征 $h^{afd}_t = [h_{which}, h_{where}, h_{shape}, h_{layout}]$
  - **[[Action Expert]]** 通过 [[Flow Matching]] / 条件扩散输出动作块 $a_{t:t+k}$
- **跨专家注意力**: [[UAA 注意力机制|UAA 渐进掩码]]，下游专家只能 query 上游 KV
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k} \in \mathbb{R}^{k \times 10}$（10-DoF 单臂位姿）

### 核心模块

#### 模块 1: Which2Act —— 对象级 grounding

**设计动机**: 在执行前先回答"应该作用在哪个物体上"。

**具体实现**:
- 用预训练 [[FLUX]] 的 [[3D Causal VAE|Flux VAE Encoder]]（冻结）把目标物体框区域编码成连续 latent $z_q \in \mathbb{R}^{C \times H \times W}$，作为目标对象的视觉表征 ground truth。
- 可供性专家输出预测 latent $\hat{z}$，与 $z_q$ 做 MSE 监督（公式 1）。
- 不用离散 codebook，避免 [[VQ-VAE]] 量化误差，得到平滑连续的"对象身份"表征。

#### 模块 2: Where2Act —— 2D 交互区域定位

**设计动机**: 回答"应该交互对象的哪个部位"。

**具体实现**:
- 一个轻量 [[Transformer]] 解码器以 affordance token + 空间位置 embedding 为输入，输出一个 $H_t \times W_t$ 的 2D affordance map。
- 用 pixel-wise [[Binary Cross-Entropy]] 监督，正样本为人工/自动标注的可交互像素（公式 2）。
- 输出会随语言指令动态变化（同一个抽屉，"open" 落在把手，"wipe" 落在面板），见 Figure 7。

#### 模块 3: How2Act —— 3D 形状 + 10-DoF 位姿布局

**设计动机**: 仅 2D 还不够，必须给动作专家提供完整 3D 几何信息（朝向、尺寸、6D 位姿 + 张开角度）。

**具体实现**:
- **Shape 分支**: 把 shape token 输入条件 [[Diffusion Model|扩散模型]]，预测一个表征 3D 形状的 latent（公式 3，噪声预测损失）。
- **Layout 分支**: 把 layout token 用 MLP 回归到 10-DoF 向量 $y_{layout} \in \mathbb{R}^{10}$（位置 3 + 旋转 6 + 夹爪 1），用 [[Smooth-L1 Loss]] 监督（公式 4）。
- 双分支结合扩散（生成性，捕捉多模态形状）和回归（确定性，保证空间精度），互补。

#### 模块 4: UAA 渐进注意力

**设计动机**: 防止"先有动作再倒推感知"的捷径，让因果链严格按 Understand → Affordance → Action 推进。

**具体实现**:
- **专家内**: 双向注意力，token 之间无掩码，最大化上下文融合。
- **跨专家**: 因果掩码——Affordance 只能 attend 到 Understanding 的 KV，Action 可以 attend 到前两者，但 Understanding 和 Affordance 都看不到 Action 的 token。
- 实现上：Affordance 用 $\text{Attention}(Q_{gen}, K_{und}, V_{und})$，Action 用 $\text{Attention}(Q_{act}, [K_{und}; K_{gen}], [V_{und}; V_{gen}])$。
- 消融显示去掉跨专家 attention（block-wise tokens）会让 LIBERO 从 95.8% 掉到 90.3%。

#### 模块 5: 三阶段训练课程

- **Stage I（可供性预训练）**: 冻结视觉编码器和动作专家，仅训练 Understanding + Affordance Expert，用 [[AGD20K]]、[[RefSpatial]]（grounding）、[[PRISM]] 412K（embodied VQA）。
- **Stage II（联合预训练）**: 解冻整个 AffordanceVLA，在 [[InternData-A1]] 大规模合成机器人数据上联合优化动作和可供性，自动标注 pipeline 生成 >100K affordance label。
- **Stage III（下游微调）**: 在目标环境的少量真实演示上微调，退火可供性权重 $\lambda_{afd}=0.15$，让动作 loss 主导。

---

## 关键公式

### 公式 1: [[Which2Act Loss|Which2Act 重建损失]]

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum \|\hat{z} - z_q\|^2
$$

**含义**: 让 Affordance Expert 预测的对象 latent $\hat{z}$ 在 [[FLUX]] VAE 潜空间内逼近 ground-truth latent $z_q$，实现对象级 grounding。

**符号说明**:
- $\hat{z} \in \mathbb{R}^{C \times H \times W}$: 模型预测的连续 latent
- $z_q \in \mathbb{R}^{C \times H \times W}$: 由冻结 Flux VAE Encoder 编码目标物体框得到的 latent
- $C, H, W$: 潜空间通道数、高、宽，用于归一化

### 公式 2: [[Where2Act Loss|Where2Act 二分类损失]]

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1 - M_i) \log(1 - \sigma(\hat{y}_i)) \right]
$$

**含义**: 像素级 [[Binary Cross-Entropy]]，监督模型预测的 affordance map 在可交互像素位置激活、其他位置抑制。

**符号说明**:
- $M_i \in \{0, 1\}$: 第 $i$ 个像素的可交互 ground truth 掩码
- $\hat{y}_i$: 模型输出 logit
- $\sigma(\cdot)$: sigmoid 激活
- $H_t, W_t$: 目标 affordance map 的空间分辨率

### 公式 3: [[How2Act Shape Loss|How2Act 形状扩散损失]]

$$
\mathcal{L}_{shape} = \mathbb{E}_{t, \varepsilon} \left[ \| \varepsilon - \hat{\varepsilon}_\theta(x_t, t, \bar{h}_{shape}) \|^2 \right]
$$

**含义**: 条件 [[Diffusion Model|扩散模型]]的 [[Denoising Loss|噪声预测损失]]，以 shape token $\bar{h}_{shape}$ 为条件生成目标物体的 3D 形状 latent。

**符号说明**:
- $\varepsilon \sim \mathcal{N}(0, I)$: 加入的高斯噪声
- $\hat{\varepsilon}_\theta$: 扩散网络预测的噪声
- $x_t$: 时刻 $t$ 的加噪样本
- $\bar{h}_{shape}$: Affordance Expert 输出的 shape 条件 token

### 公式 4: [[How2Act Layout Loss|How2Act 位姿回归损失]]

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \mathrm{SmoothL1}\left( \hat{y}^{(j)}_{layout},\ y^{(j)}_{layout} \right)
$$

**含义**: 把 10-DoF 位姿（位置 3 + 6D 旋转 + 夹爪 1）逐分量回归，[[Smooth-L1 Loss]] 兼顾鲁棒性与精度。

**符号说明**:
- $\hat{y}^{(j)}_{layout}$: 预测的第 $j$ 个位姿分量
- $y^{(j)}_{layout}$: ground-truth 位姿分量
- $\mathrm{SmoothL1}$: Huber-like 损失，在 $|x|<1$ 时为 $0.5 x^2$，否则为 $|x|-0.5$

### 公式 5: Stage I 多任务可供性目标

$$
\mathcal{L}_{\text{Stage I}} = \lambda_{which}\mathcal{L}_{which} + \lambda_{where}\mathcal{L}_{where} + \lambda_{shape}\mathcal{L}_{shape} + \lambda_{layout}\mathcal{L}_{layout}
$$

**含义**: Stage I 把四个可供性损失加权融合，专注训练 Affordance Expert 的预测能力。

**符号说明**:
- $\lambda_{which} = \lambda_{where} = \lambda_{shape} = 0.1$
- $\lambda_{layout} = 0.04$（layout 数值范围本就较小，权重下调防止主导）

### 公式 6: Stage II/III 联合动作-可供性目标

$$
\mathcal{L}_{\text{Stage II}} = \lambda_{act}\mathcal{L}_{act} + \lambda_{afd}\mathcal{L}_{afd}
$$

**含义**: 动作生成与可供性预测联合优化，可供性损失 $\mathcal{L}_{afd}$ 是公式 1-4 的加权和。

**符号说明**:
- $\lambda_{act} = 1.0$
- $\lambda_{afd} = 0.5$（Stage II），$\lambda_{afd} = 0.15$（Stage III 退火）
- $\mathcal{L}_{act}$: Action Expert 的 [[Flow Matching]] 损失

### 公式 7: UAA 跨专家注意力（Action Expert）

$$
\text{Attn}_{\text{act}} = \text{Attention}\left( Q_{act},\ [K_{und}; K_{gen}],\ [V_{und}; V_{gen}] \right)
$$

**含义**: Action Expert 既可 query Understanding，也可 query Affordance Generation 专家，形成 $U \to A \to A$ 的因果链；反之 Understanding 和 Generation 拒绝接收 Action 的 KV，从而隔绝"动作信息倒灌"。

**符号说明**:
- $Q_{act}$: Action Expert token 的 query
- $K_{und}, V_{und}$: Understanding Expert 的 KV
- $K_{gen}, V_{gen}$: Affordance Generation Expert 的 KV

---

## 关键图表

### Figure 1: 方法整体概览

![Figure 1](https://arxiv.org/html/2606.06155v1/x1.png)

**说明**: AffordanceVLA 三个专家（[[Understanding Expert|Understanding]]、[[Affordance Generation Expert|Affordance Generation]]、[[Action Expert|Action]]）协同工作，结构化可供性预测（Which/Where/How）作为感知到动作的中间表征，使方法在 [[LIBERO]]、[[CALVIN]] 仿真与真实机器人任务上同时达到 SOTA。

### Figure 2: 整体管线 + MoT 架构

![Figure 2](https://arxiv.org/html/2606.06155v1/x2.png)

**说明**: 完整 pipeline 示意。三专家共享 token 序列、独立 FFN，通过 [[UAA 注意力机制|UAA 渐进注意力]] 协调；Affordance Expert 内部分裂出 Which/Where/Shape/Layout 四类 token，分别走对应的解码头。

### Figure 3: 数据效率曲线

![Figure 3](https://arxiv.org/html/2606.06155v1/x3.png)

**说明**: 对比不同训练配置在不同下游数据规模下的成功率。AffordanceVLA 仅用 40% 训练数据即超过 vanilla [[Pi05\|π₀]] 满数据性能（LIBERO ~92%、CALVIN 链长 >4.0），验证可供性中间表征显著提升 [[Sample Efficiency|数据效率]]。

### Figure 4: 真机实验定性结果

![Figure 4](https://arxiv.org/html/2606.06155v1/x4.png)

**说明**: 基础任务（pick-place、放杯子）和复杂任务（操作抽屉、烤面包机）的真机执行序列。AffordanceVLA 在长程多步操作中保持稳健，避免 [[Pi05\|π₀]] 在复杂场景下的失败模式。

### Figure 5: 子目标 token 定量评估

![Figure 5](https://arxiv.org/html/2606.06155v1/x5.png)

**说明**: 对 affordance subgoal token 的 token accuracy 和 layout regression 误差进行综合评估，表明可供性预测本身是高质量的中间表征，并非"装饰性"辅助任务。

### Figure 6: Backbone 表征质量评估

![Figure 6](https://arxiv.org/html/2606.06155v1/x7.png)

**说明**: 随着解码器训练步数变化，骨干网络的特征提取质量评估曲线，证明加入可供性预训练 (Stage I) 强化了视觉特征的语义对齐。

### Figure 7: Where2Act 可视化

![Figure 7](https://arxiv.org/html/2606.06155v1/x8.png)

**说明**: 同一场景在不同语言指令下的 affordance map 可视化，验证 Where2Act 能动态根据指令调整可交互区域（如"open drawer"激活把手区，"clean drawer"激活面板区）。

### Table 1: LIBERO Benchmark 主结果

| Method | Spatial | Object | Goal | Long | **Average** |
|--------|---------|--------|------|------|-------------|
| [[Octo]] | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| [[OpenVLA]] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[SpatialVLA]] | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| [[CoT-VLA]] | 87.5 | 91.6 | 87.6 | 69.0 | 84.0 |
| [[Pi05\|π₀]] | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| **AffordanceVLA** | **98.6** | **98.4** | **96.2** | 89.8 | **95.8** |
| w/o Stage II | 91.4 | 88.8 | 88.0 | 76.4 | 86.2 |

**说明**: AffordanceVLA 平均 95.8%，4 项里 3 项 SOTA。Stage II 联合预训练贡献了 9.6 个百分点，证明可供性 + 动作联合训练的必要性。

### Table 2: CALVIN ABC→D 长程任务链

| Method | 1 | 2 | 3 | 4 | 5 | **Avg. Len** |
|--------|---|---|---|---|---|--------------|
| [[GR-1]] | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| Seer-Large | 96.3 | 91.6 | 86.1 | 80.3 | 73.6 | 4.28 |
| [[Pi05\|π₀]] | 95.7 | 87.5 | 80.6 | 73.0 | 47.3 | 3.84 |
| **AffordanceVLA** | **96.5** | **93.0** | **88.7** | **79.5** | **75.9** | **4.33** |
| w/o Stage II | 95.2 | 86.8 | 78.4 | 70.7 | 49.5 | 3.81 |

**说明**: AffordanceVLA 五任务链完成率 75.9%，平均链长 4.33，超过 Seer-Large（4.28）与 [[Pi05\|π₀]]（3.84）。

### Table 3: 架构消融（LIBERO Avg）

| 配置 | Avg | 影响 |
|------|-----|------|
| Frozen Affordance Expert | 67.1 | 验证可供性必须与动作联合训练 |
| w/o Which2Act | 94.6 | 对象级 grounding 重要但不致命 |
| w/o Where2Act | 93.2 | 2D 区域定位影响最大 |
| w/o How2Act | 93.7 | 3D 几何/位姿也很关键 |
| Block-wise tokens | 90.3 | 验证跨专家 attention 必要 |
| Full | **95.8** | – |

**关键发现**: 三个 affordance 模块去掉任一都掉 ~1-2.6 个点，但冻结整个 Affordance Expert 直接崩到 67.1，说明它必须在动作上下文中 **co-train** 才能学到与控制对齐的可供性。

### Table 5: 真机实验

| 任务类型 | [[Pi05\|π₀]] | **AffordanceVLA** |
|----------|-----|-------------------|
| 基础任务 (avg) | 70.8 | **88.3** |
| 复杂任务 (avg) | 44.8 | **82.9** |
| Drawer pick | 46.7 | **86.7** |
| Toaster toast | 26.7 | **86.7** |

**说明**: 复杂任务上差距最大（+38.1 个点），尤其涉及关节物体（抽屉、烤面包机）。可供性中间表征对真实世界的 OOD 泛化提供了显著帮助。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 阶段 | 用途 |
|--------|------|------|------|
| [[AGD20K]] | 20K | Stage I | 可供性 grounding 预训练 |
| [[RefSpatial]] | 250 | Stage I | 空间引用 grounding |
| [[PRISM]] | 412K | Stage I | 交互感知 VQA |
| [[InternData-A1]] | 大规模 | Stage II | 合成机器人联合训练（>100K 自动 affordance 标注） |
| [[DROID]] 子集 | – | 真机微调 | Stage III 下游 |
| [[LIBERO]] | 4 套件 | 评估 | 仿真主基准 |
| [[CALVIN]] ABC→D | 5 链 | 评估 | 长程任务 |

### 仿真主结果

- **[[LIBERO]] 平均**: AffordanceVLA **95.8%**（SOTA），相比 [[Pi05\|π₀]] 94.4% +1.4，相比 [[OpenVLA]] 76.5% +19.3。
- **[[CALVIN]] ABC→D 链长**: **4.33**（SOTA），超过 Seer-Large (4.28) 和 [[Pi05\|π₀]] (3.84)；5 步任务成功率 75.9%。

### 真机结果（Table 5）

- 基础任务（pick-place）平均 **88.3%**，远超 [[Pi05\|π₀]] 70.8%。
- 复杂任务（drawer、toaster）平均 **82.9%**，[[Pi05\|π₀]] 仅 44.8%，差距 +38.1。
- 单项最大差距：toaster toast，AffordanceVLA 86.7% vs. [[Pi05\|π₀]] 26.7%。

### 数据效率（Figure 3）

仅 40% 训练数据时：
- AffordanceVLA LIBERO ~92%，已超 vanilla [[Pi05\|π₀]] 满数据 94.4% 的水平；
- CALVIN 链长 >4.0，等价 Seer-Large 满数据成绩。

### 消融关键结论

1. **Stage II 必要**: 去掉 Stage II 联合预训练 → LIBERO 95.8 → 86.2（−9.6），CALVIN 4.33 → 3.81。
2. **Affordance Expert 必须 co-train**: 冻结 → LIBERO 直接掉到 67.1。
3. **三个 affordance 模块互补**: 单独去掉 Which / Where / How 分别掉 1.2 / 2.6 / 2.1 个点，没有"主导模块"。
4. **UAA 注意力关键**: 改成 block-wise（去掉跨专家 attention）→ 90.3，验证因果渐进注意力是核心设计。

---

## 批判性思考

### 优点

1. **可解释中间表征**: Which/Where/How 提供显式可视化中间状态，远胜端到端 VLA 的"黑盒"。
2. **数据效率显著**: 40% 数据即超过基线满数据，对真实机器人数据稀缺场景非常有用。
3. **专家分工合理**: [[MoT]] 三专家在异质任务上避免参数冲突，UAA 注意力确保因果链不被旁路。
4. **真机迁移强**: 复杂任务真机成绩较 [[Pi05\|π₀]] 提升近一倍，证明中间表征对 OOD 泛化的实际帮助。
5. **大量消融**: Stage 设计、模块、attention 类型都有完整对照实验。

### 局限性

1. **可供性标注瓶颈**: Stage II 依赖自动标注 pipeline 生成 affordance label，pipeline 本身的偏差可能传递；真实世界标注质量未充分讨论。
2. **VLM backbone 未明确**: 论文正文未指明使用哪个具体 VLM（PaliGemma / Qwen2-VL ？），不利复现。
3. **极长程任务仍受限**: 作者承认 "extremely long-horizon sequential tasks would further benefit from explicit long-term memory"，缺乏记忆模块。
4. **How2Act 绝对精度有限**: shape token accuracy 相对低（仍有效但可改进），未引入更强的几何先验（如 [[NeRF]] / [[3DGS]]）。
5. **affordance "锚定"是 hypothesis**: 作者明确说 "an interpretive hypothesis meant to guide intuition, not a claim backed by direct mechanistic evidence"——并未实证可供性 token 真的承担了感知-动作"桥梁"角色，可能只是良性正则。
6. **真机硬件信息缺**: 真实机器人平台（Franka / xArm？）正文未披露，可重复性受影响。

### 潜在改进方向

1. 引入显式长时记忆（如 [[MemoryVLA]]）扩展到极长程任务。
2. 用 [[3DGS|3D Gaussian Splatting]] 或 NeRF 表征取代 Flux VAE latent 提升 How2Act 几何精度。
3. 增加 affordance token 的可解释性诊断（probing / intervention），验证作者关于"中间表征因果作用"的假设。
4. 把 affordance 中间表征与 [[Diffusion Forcing]] / [[CoT-VLA]] 的视频/文本 CoT 结合，进一步增强可视化和可调试性。

### 可复现性评估

- [x] 代码开源（GitHub 仓库已挂）
- [ ] 预训练模型（未明确说明权重发布）
- [ ] 训练细节完整（VLM backbone、真机硬件未披露）
- [x] 数据集可获取（AGD20K / RefSpatial / PRISM / LIBERO / CALVIN 均公开）

---

## 关联笔记

### 基于
- [[Pi05|π₀]]: 直接基线和动作专家设计的灵感源
- [[MoT]]: 三专家分工的核心技术
- [[FLUX]]: 提供冻结 VAE Encoder 用于 Which2Act 监督

### 对比
- [[OpenVLA]]: 端到端 VLA 缺乏可供性中间表征
- [[CoT-VLA]]: 用未来帧 CoT；AffordanceVLA 用结构化可供性，更几何
- [[SpatialVLA]]: 也注重空间感知，但未显式拆分 Which/Where/How
- [[GR-1]]: 视频预测 CoT 路线
- [[MemoryVLA]]: 互补，可用于扩展长程

### 方法相关
- [[Affordance|可供性]]: 核心概念基础
- [[Affordance Forecasting|结构化可供性预测]]: 本文中间表征范式
- [[Understanding Expert]] / [[Affordance Generation Expert]] / [[Action Expert]]: 三专家
- [[UAA 注意力机制]]: 渐进因果跨专家注意力
- [[Flow Matching]]: 动作专家解码方式

### 数据/基准相关
- [[LIBERO]] / [[CALVIN]]: 仿真主基准
- [[AGD20K]] / [[RefSpatial]] / [[PRISM]] / [[InternData-A1]]: 训练数据
- [[DROID]]: 真机微调数据源

---

## 速查卡片

> [!summary] AffordanceVLA (PKU/HKUST-GZ/CUHK, 2026)
> - **核心**: 用 Which/Where/How 三层结构化可供性作为感知-动作中间表征
> - **架构**: MoT 三专家 + UAA 渐进因果注意力 + 三阶段课程训练
> - **结果**: LIBERO 95.8% / CALVIN 4.33 链长 / 真机复杂任务 82.9% (π₀ 仅 44.8%)
> - **亮点**: 40% 数据即超 π₀ 满数据，数据效率显著
> - **代码**: https://github.com/Skywalker-yqz/AffordanceVLA/

---

*笔记创建时间: 2026-06-05*
