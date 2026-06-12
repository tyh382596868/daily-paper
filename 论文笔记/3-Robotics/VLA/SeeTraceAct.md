---
title: "SeeTraceAct: Visibility-Aware Latent Planning from Cross-Embodiment Demonstration Videos"
method_name: "SeeTraceAct"
authors: [Jaehyeon Son, Junhyun Kim, Kyle Kam, Jeremiah Coholich, Seok Joon Kim, Jinhoo Kim, Chris Dongjoo Kim, Jaemin Cho, Dieter Fox, Zsolt Kira]
year: 2026
venue: arXiv
tags: [demo-conditioned-vla, cross-embodiment, end-effector-trace, latent-planning, robot-manipulation, flow-matching, visibility-aware]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.02745v1
created: 2026-06-12
---

# 论文笔记：SeeTraceAct: Visibility-Aware Latent Planning from Cross-Embodiment Demonstration Videos

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Georgia Institute of Technology, Allen Institute for AI (AI2), Johns Hopkins University, University of Washington |
| 日期 | June 2026 |
| 项目主页 | [GitHub](https://github.com/jaehyeon-son/SeeTraceAct) |
| 对比基线 | [[Vid2Robot]], [[UniSkill]], [[ViVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02745) / [HTML](https://arxiv.org/html/2606.02745v1) |

---

## 一句话总结

> SeeTraceAct 是一个 one-shot [[Demo-Conditioned VLA]] 框架，通过训练时的可见性感知末端执行器轨迹预测来监督视觉潜在规划，解决小目标精确定位难题，在 RoboCasa-DC 和真实机器人任务上均超越基线，平均提升真实世界成功率 12.5 个百分点。

---

## 核心贡献

1. **Visibility-Aware Trace Prediction（可见性感知轨迹预测）**: 提出在训练时对每个相机视角预测未来末端执行器轨迹坐标及其可见性标志，当末端执行器离开某视角时不强制产生无效监督，同时用 BCE 损失训练可见性预测头，避免 ill-posed 训练目标。
2. **Visual Latent Plan（视觉潜在规划）**: 给定演示视频和当前观测，通过 [[V-JEPA]] + [[Resampler|Perceiver Resampler]] 编码出任务级别的视觉潜在规划 $z$；训练时通过轨迹解码器提供额外空间定位监督，推理时直接从 $z$ 生成动作，无需解码器参与。
3. **RoboCasa-DC Benchmark**: 提出并开源 RoboCasa-DC——RoboCasa 的 [[Demo-Conditioned VLA|demo-conditioned]] 扩展版，提供 Franka Panda 轨迹与对应 GR-1 人形机器人演示的配对数据（24 个任务，每任务 100 条配对轨迹、50 条评测 seed），支持同体 / 跨体演示两种评测模式。

---

## 问题背景

### 要解决的问题

One-shot [[Demo-Conditioned VLA]] 面临核心难题：给定单次演示视频（可能来自不同机体），机器人需要精确地定位并操作小型目标（如咖啡机按钮、水龙头旋钮）。现有端到端方法在需要亚厘米级精确定位的任务（precision-sensitive tasks）上容易失败。

### 现有方法的局限

- **端到端 demo-conditioned VLA**（如 [[GR00T N1.5]]）：缺乏显式的空间定位监督，难以在小目标上精确落点。
- **[[TraceVLA]]** 等轨迹提示方法：将轨迹作为输入提示，但未解决多视角场景下末端执行器部分可见时的监督问题；且依赖推理时的显式轨迹输入。
- **跨体差距**（cross-embodiment gap）：演示视频来自不同机器人（如 GR-1 人形）或人类，直接将演示动作映射到目标机器人在空间上本质是 ill-posed 的。

### 本文的动机

核心洞察：即使演示来自不同机体，**末端执行器在相机图像平面中的运动轨迹**仍携带有效的空间定位信息。训练时用这一信息作为辅助监督信号，可以迫使潜在规划 $z$ 捕获精细的空间细节；推理时再将 $z$ 直接解码为动作，不依赖外部轨迹输入，实现精准操作。

---

## 方法详解

### 模型架构

SeeTraceAct 名称对应三个阶段：**See**（视觉编码）→ **Trace**（轨迹预测，仅训练）→ **Act**（动作生成）：

- **输入**: 语言指令 $l$ + 演示视频 $D$ + 当前观测 $o_t$ + 机器人状态 $q_t$
- **演示视频编码**: [[V-JEPA]] 2 encoder → [[Resampler|Perceiver Resampler]]，将 8192 个视频 token 压缩为 32 个 token $\psi_t$
- **Backbone**: [[GR00T N1.5]]（预训练 VLA 骨干）
- **核心模块**:
  - **Visual Latent Plan**: 在输入 token 之后追加可学习的 query token，在因果注意力机制下，这些 query token 的最终隐状态形成视觉潜在规划 $z_t$
  - **[[Visibility-Aware Trace Loss|Visibility-Aware Trace Decoder]]**: 训练时从 $z_t$ 解码每视角末端执行器坐标 $\hat{y}$ 和可见性 $\hat{m}$
  - **Action Expert**: 从 $\phi_t$、$\psi_t$、$z_t$ 和噪声动作块 $A_t^\tau$ 中预测速度场，用 [[Flow Matching]] 去噪
- **输出**: 动作块 $A_t$（$H=16$ 步），推理时以 $K=4$ 步 flow-matching 去噪
- **推理简化**: 推理阶段完全丢弃 Trace Decoder，仅保留 See + Act 两步，无额外推理开销

### 核心模块

#### 模块一：Video Encoding（视频编码）

**设计动机**: 演示视频包含任务执行的时序语义信息，但原始视频帧 token 数量庞大（[[V-JEPA]] 编码 64 帧后为 8192 个 token），需要有效压缩。

**具体实现**:
- [[V-JEPA]] 2 对演示视频 $D$ 的 64 帧提取特征，输出 8192 个 token
- [[Resampler|Perceiver Resampler]] 将 8192 个 token 压缩为 32 个视频 token $\psi_t$
- 在 VLA 主干网络中，$\psi_t$ 与语言和图像 token 拼接后统一输入

#### 模块二：Visual Latent Plan（视觉潜在规划）

**设计动机**: 需要一个紧凑的任务级表示，在训练时接受来自轨迹解码器的额外梯度，在推理时直接驱动动作生成，形成有效的信息瓶颈。

**具体实现**:
- 在输入序列末尾追加若干可学习的 query token
- 在 [[GR00T N1.5]] 骨干的因果自注意力下，这些 query token 同时 attend 到图像、语言和演示视频 token
- 最终 query token 隐状态构成视觉潜在规划 $z_t \in \mathbb{R}^{d}$，作为动作专家和轨迹解码器的共同输入

#### 模块三：Visibility-Aware Trace Decoder（可见性感知轨迹解码器）

**设计动机**: 机器人通常配置多个相机（腕部摄像头、外部摄像头），末端执行器在某些视角中会出现部分或完全不可见的情况，强制在这些视角预测坐标会引入噪声梯度。

**具体实现**:
- **Regression head**：预测归一化 2D 末端执行器坐标 $\hat{y}_{t+n\Delta} \in [0,1]^2$，每隔 $\Delta=5$ 帧预测一次，共预测 $N=5$ 个点
- **Validity head**：预测每个轨迹点是否在图像边界内的二值标志 $\hat{m}_{t+n\Delta} \in \{0,1\}$
- 真实标签 $y_{t+n\Delta}$ 通过正向运动学 + 相机投影矩阵从机器人状态计算；$m_{t+n\Delta}$ 通过判断投影坐标是否在 $[0,1]^2$ 范围内确定
- **仅训练时启用**，推理时整个 Trace Decoder 被丢弃

#### 模块四：Action Expert（动作专家）

**设计动机**: 采用 [[Flow Matching]] 生成高质量连续动作序列，相比回归损失更能建模多模态动作分布。

**具体实现**:
- 接收图像特征 $\phi_t$、视频特征 $\psi_t$、潜在规划 $z_t$、机器人状态 $q_t$ 和带噪动作块 $A_t^\tau$
- 预测速度场 $V_\theta$，通过 $K=4$ 步 flow-matching ODE 从纯噪声去噪到动作
- 动作块 $A_t \in \mathbb{R}^{H \times d_a}$，$H=16$，$d_a$ 为动作维度（6-DoF EE 增量 + 夹爪）

---

## 关键公式

### 公式 1：[[Flow Matching|动作预测损失（Flow Matching）]]

$$
\mathcal{L}_{\text{act}}(\theta) = \mathbb{E}_{\tau,\epsilon}\left[\left\|V_{\theta}(\phi_t, \psi_t, z_t, A_t^{\tau}, q_t) - (A_t - \epsilon)\right\|^2\right]
$$

**含义**: 训练 Flow Matching 速度场预测器 $V_\theta$，预测目标为从噪声动作 $A_t^\tau$ 到真实动作 $A_t$ 的方向向量。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 流匹配时间步
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $A_t^{\tau} = \tau A_t + (1-\tau)\epsilon$: 在真实动作和纯噪声之间插值的带噪动作块
- $V_\theta$: 速度场预测网络
- $\phi_t$: 当前相机观测特征
- $\psi_t$: 演示视频 token（经过 [[Resampler|Perceiver Resampler]] 压缩后的 32 个 token）
- $z_t$: 视觉潜在规划向量
- $q_t$: 机器人关节状态

### 公式 2：[[Visibility-Aware Trace Loss|轨迹坐标回归损失]]

$$
\mathcal{L}_{\text{reg}}(\theta) = \frac{1}{\max(1, \sum_{n=1}^N m_{t+n\Delta})} \sum_{n=1}^N m_{t+n\Delta} \|\hat{y}_{t+n\Delta} - y_{t+n\Delta}\|_1
$$

**含义**: 仅在末端执行器真实可见（$m_{t+n\Delta}=1$）的时刻计算轨迹坐标 L1 回归损失，并以可见帧数归一化，防止末端执行器大部分时间不可见时损失被过度稀释。

**符号说明**:
- $N=5$: 预测的轨迹点数
- $\Delta=5$: 相邻轨迹点的时间间隔（帧数）
- $m_{t+n\Delta} \in \{0,1\}$: 第 $t+n\Delta$ 帧的真实可见性标签
- $\hat{y}_{t+n\Delta} \in [0,1]^2$: 预测的归一化 2D 末端执行器坐标
- $y_{t+n\Delta}$: 真实末端执行器坐标（由正向运动学 + 相机投影计算）
- $\max(1, \cdot)$: 防止分母为零（当所有帧均不可见时）

### 公式 3：[[Visibility-Aware Trace Loss|可见性预测损失]]

$$
\mathcal{L}_{\text{valid}}(\theta) = \frac{1}{N} \sum_{n=1}^N \mathrm{BCE}(\hat{m}_{t+n\Delta}, m_{t+n\Delta})
$$

**含义**: 对所有 $N$ 个轨迹点的可见性预测计算二元交叉熵损失，不受可见性掩码影响，确保模型学会预测末端执行器何时不在视野内。

**符号说明**:
- $\hat{m}_{t+n\Delta}$: 可见性预测 logit
- $m_{t+n\Delta}$: 真实可见性标签
- $\mathrm{BCE}(\cdot, \cdot)$: 二元交叉熵

### 公式 4：[[Visibility-Aware Trace Loss|联合轨迹损失]]

$$
\mathcal{L}_{\text{trace}}(\theta) = \mathcal{L}_{\text{reg}}(\theta) + \lambda_{\text{valid}} \mathcal{L}_{\text{valid}}(\theta)
$$

**含义**: 轨迹总损失由坐标回归损失和可见性预测损失加权求和，$\lambda_{\text{valid}}=1.0$。

**符号说明**:
- $\lambda_{\text{valid}} = 1.0$: 可见性损失权重

### 公式 5：[[Visibility-Aware Trace Loss|总训练目标]]

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{act}}(\theta) + \lambda_{\text{trace}} \mathcal{L}_{\text{trace}}(\theta)
$$

**含义**: 最终训练目标为动作 [[Flow Matching]] 损失与可见性感知轨迹损失的加权求和，两路监督共同作用于视觉潜在规划 $z_t$。

**符号说明**:
- $\lambda_{\text{trace}} = 0.2$: 轨迹损失权重（通过消融实验确定）

### 公式 6：[[Flow Matching|Flow Matching 推理过程]]

$$
A_t^{(k+1)/K} = A_t^{k/K} + \frac{1}{K} V_\theta\!\left(\phi_t, \psi_t, z_t, A_t^{k/K}, q_t\right), \quad k = 0, 1, \ldots, K-1
$$

**含义**: 推理时从纯噪声 $A_t^0 \sim \mathcal{N}(0,I)$ 出发，以 $K=4$ 步欧拉积分沿速度场方向去噪，最终得到动作块 $A_t^1$。

**符号说明**:
- $K=4$: 推理去噪步数
- $A_t^0 \sim \mathcal{N}(0,I)$: 初始纯噪声动作块

---

## 关键图表

### Figure 1：SeeTraceAct 整体框架

![Figure 1 - SeeTraceAct Overview](https://arxiv.org/html/2606.02745v1/x1.png)

**说明**: SeeTraceAct 的三阶段流程示意。给定演示视频、当前多视角图像和语言指令，编码器生成视觉潜在规划 $z_t$（**See**）。训练时，轨迹解码器从 $z_t$ 解码多视角末端执行器轨迹及其可见性（**Trace**）。动作专家从 $z_t$ 通过 [[Flow Matching]] 预测动作块（**Act**）。推理时轨迹解码器被完全丢弃。

### Figure 2：模型架构细节

![Figure 2 - Architecture](https://arxiv.org/html/2606.02745v1/x2.png)

**说明**: 完整架构图，展示四路输入流（相机图像 $\phi_t$、语言指令 $l$、演示视频 $\psi_t$、机器人状态 $q_t$）经 [[GR00T N1.5]] VLA 骨干处理后，可学习 query token 的最终隐状态构成 $z_t$，分别连接训练专用的轨迹解码器（Regression head + Validity head）和动作专家。

### Figure 3：RoboCasa-DC 数据集构建

![Figure 3 - RoboCasa-DC Dataset](https://arxiv.org/html/2606.02745v1/x3.png)

**说明**: RoboCasa-DC 的构成示意。对每个 24 个任务，收集 100 条 Franka Panda 轨迹与对应 GR-1 人形机器人演示视频的配对训练数据；评测时为 50 条预定义 seed 的 Panda 轨迹与对应人形演示，形成同体（Panda←Panda）和跨体（Panda←GR-1）两种评测模式。

### Figure 4：真实世界任务（已见/未见）

![Figure 4 - Real-World Tasks A](https://arxiv.org/html/2606.02745v1/x4.png)
![Figure 4b - Real-World Tasks B](https://arxiv.org/html/2606.02745v1/x5.png)

**说明**: (a) 四个已见任务和 (b) 四个未见任务的真实世界 benchmark 场景。黄色箭头指示末端执行器期望运动路径，覆盖抓取、放置、按键等多种操作类型。

### Figure 5：真实世界实验结果

![Figure 5 - Real-World Results](https://arxiv.org/html/2606.02745v1/x6.png)

**说明**: Franka Panda 真实机器人实验成功率对比（每任务 10 次 trial）。SeeTraceAct 在全部四个未见任务上均达到最优，平均成功率 50%，比最强基线 ViVLA（40%）高 10 个百分点。

### Figure 6：硬件配置

![Figure 6 - Hardware Setup](https://arxiv.org/html/2606.02745v1/x7.png)

**说明**: 真实世界实验硬件配置。Franka Panda 机械臂配备外部第三人称视角相机和腕部摄像头（$K=2$ 视角），腕部摄像头视野中末端执行器频繁不可见，直接体现了可见性感知设计的必要性。

### Figure 7：精度敏感任务目标区域可视化

![Figure 7 - Target Interaction Regions](https://arxiv.org/html/2606.02745v1/x8.png)

**说明**: RoboCasa-DC 各任务中机器人需要交互的目标区域（高亮标注）。Target Interaction Ratio（TIR）越小表示目标越小、任务精度要求越高。SeeTraceAct 在 TIR 小的任务上相对提升更显著（$\rho = -0.80$，$p < 10^{-5}$），验证了空间定位假设。

### Table 1：RoboCasa-DC 仿真评测结果

| 方法 | 同体-类别均衡 | 跨体-类别均衡 | 同体-精度敏感 | 跨体-精度敏感 |
|------|:---:|:---:|:---:|:---:|
| Vid2Robot | 21.5% | 8.8% | 12.6% | 6.4% |
| UniSkill | 13.3% | 11.2% | 10.4% | 6.0% |
| ViVLA | 14.4% | 8.0% | 8.9% | 8.4% |
| **SeeTraceAct（本文）** | **23.0%** | **11.6%** | **14.1%** | **12.8%** |

**关键发现**: SeeTraceAct 在全部四个设置上均取得最优成功率。在精度敏感的跨体设置（最难）上以 12.8% 相比 ViVLA 的 8.4% 优势最为显著（+52%），印证了可见性感知轨迹监督对小目标精确定位的贡献。

### Table 2：真实世界实验结果

| 任务 | Vid2Robot | UniSkill | ViVLA | SeeTraceAct |
|------|:---------:|:--------:|:-----:|:-----------:|
| Pick Block | 50% | 40% | 40% | **60%** |
| Pick Cup | 30% | 20% | 40% | **50%** |
| Press Button | 40% | 50% | 40% | **50%** |
| Stack (Swapped) | 30% | 30% | 40% | **40%** |
| **平均** | 37.5% | 35% | 40% | **50%** |

**关键发现**: SeeTraceAct 平均成功率 50%，比最强基线 ViVLA（40%）提升 12.5 个百分点，所有任务上均达到最优或并列最优。

### Table 3：消融实验（跨体-类别均衡设置）

| 配置 | 成功率 |
|------|:------:|
| GR00T N1.5 基线（无 demo 条件） | 9.0% |
| w/o 轨迹监督（仅演示视频编码） | 9.4% |
| w/o validity head（全时刻计算坐标损失） | 8.2% |
| w/o action-aware 视频编码 | 9.2% |
| w/ 3D 轨迹监督 | 10.4% |
| **完整 SeeTraceAct** | **12.2%** |

**关键发现**: 移除 validity head 导致性能最大幅度下降（9.4% → 8.2%），反而低于无轨迹监督基线，直接验证了可见性感知设计的必要性——在不处理可见性问题的情况下强制轨迹监督是有害的。3D 轨迹监督（10.4%）也略逊于完整模型（12.2%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboCasa-DC（仿真） | 24 任务 × 100 条/任务 Panda 轨迹 + GR-1 演示配对 | 跨体配对；两种评测分割（类别均衡/精度敏感） | 训练 + 仿真评测 |
| 真实世界 Franka 数据集 | 4 已见 + 4 未见任务，每任务 10 次 trial | Franka Panda + 人类手部演示视频 | 真实世界评测 |

**评测分割**:
- **类别均衡（Category-Balanced）**: 5 个多样性未见任务（CloseDrawer, TurnOffSinkFaucet, OpenDoubleDoor, CoffeeServeMug, PnPCounterToMicrowave）
- **精度敏感（Precision-Sensitive）**: 5 个小目标操作任务（TurnOffStove, CoffeePressButton, TurnOffSinkFaucet, PnPCounterToSink, PnPStoveToCounter）

### 实现细节

- **Backbone**: [[GR00T N1.5]]（NVIDIA 预训练 VLA 骨干）
- **演示视频编码**: [[V-JEPA]] 2 + [[Resampler|Perceiver Resampler]]（64 帧 → 8192 → 32 token）
- **动作空间**: 末端执行器 6-DoF 增量 + 夹爪二值状态
- **动作预测**: [[Flow Matching]]，训练时均匀采样 $\tau$，推理时 $K=4$ 步欧拉积分
- **动作 Horizon**: $H=16$
- **轨迹参数**: $N=5$ 个预测点，步长 $\Delta=5$ 帧
- **损失权重**: $\lambda_{\text{valid}}=1.0$，$\lambda_{\text{trace}}=0.2$
- **优化器**: AdamW，学习率 $10^{-4}$
- **训练步数**: 50,000 步，批大小 128
- **硬件**: NVIDIA GPU 集群（训练）；Franka Panda 实时控制器（推理，$K=2$ 相机）

### 精度相关性分析

对 TIR（Target Interaction Ratio，目标区域面积占图像比例）和 SeeTraceAct 相对于基线的提升之间计算 Spearman 相关系数：

| 对比基线 | Spearman ρ | p 值 |
|---------|:----------:|:----:|
| vs. Vid2Robot | -0.80 | < 10⁻⁵ |
| vs. UniSkill | -0.45 | < 0.05 |
| vs. ViVLA | -0.52 | < 0.01 |

TIR 越小（目标越精细），SeeTraceAct 提升越大，强有力地支持了"轨迹监督改善精确空间定位"的核心假设。

---

## 批判性思考

### 优点

1. **优雅的信息瓶颈设计**: 轨迹解码器仅在训练时存在，推理时零开销，实现了"用轨迹监督提升空间感知但不依赖轨迹执行"的解耦。
2. **可见性感知的务实设计**: 消融实验明确证明忽略可见性会损害性能（8.2% < 9.4%），而解决可见性问题是关键的正向贡献。
3. **完整的 Benchmark 贡献**: RoboCasa-DC 同时提供同体/跨体两种模式和精度/类别两个维度的评测分割，极大提升了 demo-conditioned VLA 的可比较性。
4. **精度相关性的量化验证**: 通过 TIR 指标和相关系数定量验证方法的作用场景，而非仅依赖平均成功率。

### 局限性

1. **轨迹标注依赖正向运动学 + 相机标定**: 需要精确的机器人 URDF 模型和相机外参，部署成本较高。
2. **演示配对假设**: 需要同一任务的演示视频与执行轨迹配对，在开放世界数据采集中难度较大。
3. **真实世界评测规模有限**: 仅在单臂桌面操作 benchmark 上测试，缺乏双臂、灵巧手或移动操作场景。
4. **绝对成功率偏低**: RoboCasa-DC 上最优成功率约 12-23%，说明 demo-conditioned 机器人学习整体仍有很大提升空间。

### 潜在改进方向

1. **3D 轨迹监督**: 消融显示 3D 轨迹（10.4%）与 2D 轨迹（12.2%）接近但更难获取，结合深度信息或许能进一步提升。
2. **无监督演示配对**: 探索无需配对轨迹标注的自监督轨迹目标，降低数据采集成本。
3. **多演示融合**: 从 one-shot 扩展至 few-shot，聚合多个演示的潜在规划有望进一步提升泛化精度。

### 可复现性评估

- [x] 代码开源（GitHub: [jaehyeon-son/SeeTraceAct](https://github.com/jaehyeon-son/SeeTraceAct)）
- [x] 数据集开源（RoboCasa-DC benchmark）
- [x] 训练细节完整（消融实验详细，超参数公开）
- [x] 数据集可获取（RoboCasa 基础仿真环境开源）

---

## 关联笔记

### 基于

- [[TraceVLA]]: 视觉轨迹提示的前置工作，SeeTraceAct 将轨迹从输入端提示移至训练辅助目标
- [[GR00T N1.5]]: 所有方法（含基线）统一采用的 VLA 骨干，实现公平对比
- [[V-JEPA]]: 演示视频编码器，提供丰富的视觉时序特征
- [[Action Chunking]]: 动作块预测的基础设计
- [[Flow Matching]]: 动作生成的核心范式

### 对比

- [[Vid2Robot]]: SigLIP + Perceiver Resampler + 视频-视频对比损失，RoboCasa-DC 最强同体基线
- [[UniSkill]]: Inverse Skill Dynamics 模块，skill interval k=20
- [[ViVLA]]: V-JEPA 2 + LAPA 潜在动作编码器，8 个 LACT token

### 方法相关

- [[Visibility-Aware Trace Loss]]: 核心技术创新，可见性感知轨迹损失
- [[Demo-Conditioned VLA]]: 任务设定范式
- [[Cross-Embodiment Learning]]: 跨体学习的核心挑战

### 硬件/数据相关

- [[Franka Panda]]: 真实实验平台
- [[GR00T N1.5|GR-1 人形机器人]]: 演示来源机体（RoboCasa-DC 跨体演示）
- [[RoboCasa-DC]]: 本文提出并开源的仿真 benchmark

---

## 速查卡片

> [!summary] SeeTraceAct (arXiv 2606.02745)
> - **核心**: One-shot [[Demo-Conditioned VLA]]，通过可见性感知末端执行器轨迹预测监督视觉潜在规划，提升小目标空间定位精度
> - **方法**: See（[[V-JEPA]] 2 + Perceiver Resampler → 视觉潜在规划 $z_t$）→ Trace（训练时用 Regression + Validity 双头解码多视角轨迹）→ Act（[[Flow Matching]] 生成动作块，推理时丢弃 Trace）
> - **数据集**: 开源 [[RoboCasa-DC]]（24 任务，Panda-GR-1 跨体配对，100 条训练/50 条评测）
> - **结果**: RoboCasa-DC 四设置全优（最优 23.0%）；真实世界平均成功率 50%（+12.5pp vs 最强基线）
> - **代码**: https://github.com/jaehyeon-son/SeeTraceAct

---

*笔记创建时间: 2026-06-12*
