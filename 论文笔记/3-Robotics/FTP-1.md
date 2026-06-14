---
title: "FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation"
method_name: "FTP-1"
authors: [Chengbo Yuan, Zicheng Zhang, Mingjie Zhou, Wendi Chen, Yi Wang, Zhuoyang Liu, Dantong Niu, Shuo Wang, Hui Zhang, Wenkang Zhang, Yingdong Hu, Yuanqing Gong, Wanli Xing, Chuan Wen, Cewu Lu, Kaifeng Zhang, Yang Gao]
year: 2026
venue: arXiv
tags: [tactile-policy, foundation-model, contact-rich-manipulation, heterogeneous-sensors, transfer-learning]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.13102
created: 2026-06-14
---

# 论文笔记：FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, Shanghai Qi Zhi Institute, Sharpa, Shanghai Jiao Tong University, UC Berkeley, ETH Zurich |
| 日期 | June 2026 |
| 项目主页 | [ftp1-policy.github.io](https://ftp1-policy.github.io/) |
| 对比基线 | [[π0.5]]、[[Tactile-VLA]]、[[VITaL]]、[[UniVTAC-ACT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13102) |

---

## 一句话总结

> FTP-1 是首个跨触觉传感器的通用基础触觉策略，通过形态感知的统一 Token 空间和大规模异构数据预训练，在已见传感器上提升 17.2%、在未见传感器上提升 31% 的接触丰富操作成功率。

---

## 核心贡献

1. **首个通用触觉基础策略**: 提出 FTP-1，首次实现跨 21 种触觉传感器（图像/数组/状态三类）的统一策略预训练，并能迁移到预训练期间未见过的传感器
2. **形态感知触觉 Token 空间（MTTS）**: 设计 24 个功能区 slot 的统一接口，将不同模态、不同传感器的触觉信号统一编码，配合可学习功能区嵌入实现传感器无关的表征
3. **大规模异构触觉数据集（FTP-1-Dataset）**: 汇聚 26 个数据源、21 种传感器、约 3000 小时的触觉操作数据，覆盖人类示范、灵巧手机器人、夹爪机器人三类数据分布

---

## 问题背景

### 要解决的问题

接触丰富操作（contact-rich manipulation）依赖触觉感知，但现有触觉策略高度传感器特异性——每种传感器输出格式（图像/数组/力矩状态）不同，机器人本体各异，导致数据无法跨配置复用，策略也无法迁移。

### 现有方法的局限

- 基于[[视觉语言动作模型|VLA]]的通用策略（如 [[π0.5]]）已实现视觉跨任务泛化，但大多忽略触觉输入
- 现有触觉策略（如 [[VITaL]]、[[UniVTAC-ACT]]）绑定特定传感器配置，难以跨本体迁移
- [[表征预训练]]方法（如 T3）改善了下游性能，但尚未建立端到端的传感器无关策略

### 本文的动机

类比视觉基础模型（ViT 等）通过统一 patch token 消除图像尺寸差异，FTP-1 设计形态感知的 token 化方案，将不同触觉传感器的异构信号统一映射到同一语义空间，进而利用大规模跨传感器数据实现策略级知识迁移。

---

## 方法详解

### 模型架构

![Figure 2: FTP-1 整体架构](https://arxiv.org/html/2606.13102v1/x2.png)

**说明**: FTP-1 总体框架。异构触觉观测经过传感器特异编码器映射到统一的 [[MTTS|触觉 Token 空间]]，送入共享的触觉专家 Transformer，再与视觉语言和本体感知信息融合，由[[流匹配|Flow-Matching]] 动作专家生成动作块输出。

FTP-1 采用**多专家融合**架构：

- **输入**: 语言指令 $\ell$ + 多视角 RGB 观测 $\mathcal{I}_t$ + 本体感知 $\mathbf{s}_t$ + 触觉观测 $\mathcal{X}_t$
- **视觉语言专家**: 基于 [[π0.5]] 初始化的视觉语言 Transformer
- **触觉专家**: 300M 参数共享 Transformer，处理 MTTS 统一 token
- **动作专家**: [[流匹配]] 动作专家，单向 attend 触觉专家（触觉专家不反向 attend 动作专家）
- **输出**: [[动作块|Action Chunk]] $\hat{\mathbf{A}}_{t:t+H-1} \in \mathbb{R}^{H \times D}$

与基于适配器注入的方法不同，FTP-1 使用**独立触觉专家**设计：触觉专家是独立的 Transformer 分支，不通过 adapter 嵌入主干，保持触觉表征的完整性。

### 核心模块

#### 模块 1: 形态感知触觉 Token 空间（MTTS）

**设计动机**: 不同触觉传感器的物理安装位置对应手部相同的功能区域（指尖、掌心、腕部等），利用[[功能区嵌入]]将物理意义对齐，实现传感器无关的语义 token。

**24 个功能区 slot 定义**:

![Figure 3: MTTS 功能区定义](https://arxiv.org/html/2606.13102v1/figure/definition_tactile_torque_function_area.png)

- **Slots 0–14**: 手部不同功能区域（指尖、指腹、掌心等）
- **Slots 15–20**: 腕部/手指的力/力矩（force/torque）信号
- **Slots 21–23**: 预留给未来新传感器类型
- **夹爪映射**: 双指夹爪传感器分别映射到拇指尖（slot 0）和食指尖（slot 1）

**可学习功能区嵌入**: 每个 slot 对应一个可学习嵌入向量，在送入触觉专家前叠加到 token 上；左右手用独立嵌入区分。

#### 模块 2: 异构触觉编码器

**设计动机**: 三类触觉传感器信号格式差异巨大，需传感器特异的轻量前端将各自模态压缩为统一维度 token，再送入共享后端。

**三类编码器**:

| 传感器类型 | 典型传感器 | 编码方式 | 输出 |
|-----------|-----------|---------|------|
| **图像型** | GelSight | 轻量传感器特异 ViT + 共享预训练 [[T3]] Transformer | 取 `[CLS]` token |
| **数组型** | Contactile | CNN 捕获空间结构，每个功能区压缩为一个 token | 每区一 token |
| **状态型** | 力/力矩传感器 | [[Fourier 编码]] + 轻量 MLP | 功能区 token |

#### 模块 3: 自适应 RMSNorm 本体感知注入器（Adaptive RMSNorm Proprioception Injector）

**设计动机**: 将本体感知状态 $\mathbf{s}_t$ 以条件化方式注入触觉专家，改善性能与泛化能力。

**具体实现**: 基于 [[π0.5]] 架构的 Adaptive [[RMSNorm]]，将本体感知作为缩放/偏移条件，与触觉 token 一同送入 Transformer 层归一化。

---

## 关键公式

### 公式 1: [[策略函数|动作预测]]

$$
\hat{\mathbf{A}}_{t:t+H-1} = \pi_{\theta}(\ell,\, \mathcal{I}_{t},\, \mathbf{s}_{t},\, \mathcal{X}_{t})
$$

**含义**: FTP-1 策略接收语言、视觉、本体感知和触觉四类输入，预测未来 $H$ 步的动作块。

**符号说明**:
- $\hat{\mathbf{A}}_{t:t+H-1} \in \mathbb{R}^{H \times D}$: 预测的动作块，$H$ 为动作视野，$D$ 为统一动作空间维度
- $\ell$: 语言指令
- $\mathcal{I}_t$: 多视角 RGB 观测
- $\mathbf{s}_t$: 本体感知（关节角度、末端位姿等）
- $\mathcal{X}_t$: 触觉观测（来自当前配置的各传感器信号集合）
- $\pi_{\theta}$: FTP-1 策略网络参数

---

## 关键图表

### Figure 1: FTP-1 概览

![Figure 1: FTP-1 概览与核心结果](https://arxiv.org/html/2606.13102v1/x1.png)

**说明**: FTP-1 是首个通用触觉基础策略，在大规模异构触觉操作数据集上预训练，在未见传感器配置上实现 31.6% 的成功率提升。

### Figure 2: FTP-1 架构总览

![Figure 2: FTP-1 整体架构](https://arxiv.org/html/2606.13102v1/x2.png)

**说明**: 异构触觉观测经传感器特异编码器 → [[MTTS]] 统一 token 空间 → 共享触觉专家 → 与视觉语言和本体感知融合 → [[流匹配]] 动作专家生成输出。

### Figure 3: MTTS 功能区定义

![Figure 3: MTTS 触觉功能区定义](https://arxiv.org/html/2606.13102v1/figure/definition_tactile_torque_function_area.png)

**说明**: 24 个功能区 slot 在手部解剖结构上的分布，slots 0–14 覆盖手部触觉区域，slots 15–20 对应力/力矩信号，slots 21–23 预留扩展。

### Figure 4: FTP-1-Dataset 概览

![Figure 4: FTP-1-Dataset 数据集构成](https://arxiv.org/html/2606.13102v1/x3.png)

**说明**: 数据集汇聚 26 个来源、21 种触觉传感器（7 图像型、5 数组型、9 状态型），约 20% 人类数据、30% 灵巧手机器人、50% 夹爪机器人，均通过 MTTS 接口统一表示。

### Figure 5: 已见传感器微调实验

![Figure 5: 已见传感器微调实验概览](https://arxiv.org/html/2606.13102v1/x4.png)

**说明**: 对预训练期间已见的传感器配置进行微调实验设置，覆盖仿真 UniVTAC 基准和真实机器人（Sharpa North + Sharpa&Dexmate）任务。

### Figure 6: 未见传感器迁移实验

![Figure 6: 未见传感器迁移实验概览](https://arxiv.org/html/2606.13102v1/x5.png)

**说明**: 对预训练期间从未见过的新传感器配置（FlexivXense 和 TactileUMI）进行零样本/少样本微调实验，验证 FTP-1 的跨传感器知识迁移能力。

### Figure 7: FTP-1 vs NTP-1 消融对比

![Figure 7: FTP-1 与 NTP-1 对比](https://arxiv.org/html/2606.13102v1/figure/ntp_experiment_result.png)

**说明**: NTP-1（无触觉预训练对照）在已见传感器（UniVTAC）上虽优于 FTP-π₀.₅，说明数据分布对齐有效；但在未见传感器（FlexivXense）上，FTP-1 比 NTP-1 高出 **37.5%**，确认触觉专家预训练是跨传感器迁移的关键。

### Table 1: UniVTAC 仿真结果（已见传感器）

| Method | Lift Bottle | Pull-out Key | Lift Can | Put Bottle | Insert Hole | Insert Tube | Avg. | Avg. w/o Lift |
|--------|-------------|--------------|----------|------------|-------------|-------------|------|----------------|
| VITaL | 72 | 47 | 8 | 32 | 25 | 34 | 36.33 | 34.5 |
| UniVTAC-ACT | 71 | 46 | 29 | 31 | 25 | 56 | 43.00 | 39.5 |
| π₀.₅ | 97 | 38 | 72 | 16 | 31 | 41 | 49.16 | 31.5 |
| Tactile-VLA | 97 | 32 | 15 | 10 | 41 | 56 | 41.83 | 34.75 |
| FTP-π₀.₅ | 77 | 30 | 26 | 19 | 47 | 72 | 45.16 | 42.0 |
| **FTP-1** | **97** | **48** | **65** | **47** | **64** | **79** | **66.66** | **59.5** |

**关键发现**: FTP-1 以 66.66% 平均成功率大幅领先所有基线，比第二名 FTP-π₀.₅ 高出 21.5 个百分点，在 Put Bottle（接触稳定性要求高）和 Insert Hole/Tube（精确接触要求高）任务上优势尤为显著。

### Table 2: 真实机器人结果（已见传感器）

| Method | Draw Balloon | Fix Hand (Tear) | Fix Hand (Finish) | Twist Cap | Flip Book | Wipe Dish | Average |
|--------|--------------|-----------------|-------------------|-----------|-----------|-----------|---------|
| π₀.₅ | 35 | 70 | 35 | 40 | 65 | 30 | 45.3 |
| Tactile-VLA | 20 | 80 | 25 | 10 | 45 | 35 | 35.8 |
| FTP-π₀.₅ | 25 | 65 | 25 | 20 | 70 | 45 | 41.6 |
| **FTP-1** | **45** | **80** | **40** | **65** | **85** | **60** | **62.5** |

**关键发现**: 真实机器人场景下，FTP-1 在 Twist Cap（需精确力控）和 Flip Book（需感知纸张）等高接触精度任务上表现突出，平均成功率 62.5%，比 π₀.₅ 高 17.2 个百分点。

### Table 3: 未见传感器迁移结果

| Method | Insert Hanoi | Insert USB | Wipe Board | Average |
|--------|--------------|-----------|-----------|---------|
| π₀.₅ | 25 | 0 | 20 | 15.0 |
| Tactile-VLA | 0 | 10 | 15 | 8.3 |
| FTP-π₀.₅ | 5 | 10 | 30 | 15.0 |
| **FTP-1** | **55** | **30** | **55** | **46.6** |

**关键发现**: 在预训练期间从未见过的 FlexivXense + TactileUMI 传感器上，FTP-1 以 46.6% 的平均成功率大幅领先所有基线（最强基线仅 15%），凸显 MTTS 的跨传感器泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 传感器类型 | 用途 |
|--------|------|-----------|------|
| FTP-1-Dataset（汇聚） | 26 来源，~3000 小时 | 21 种（7 图像/5 数组/9 状态型）| 预训练 |
| Sharpa North-FTP-1（新增） | 4000 条长视野示范 | Sharpa North 触觉手 | 预训练+测试 |
| UniVTAC | 仿真基准 | GelSight 类 | 仿真评估 |
| FlexivXense | 真实机器人 | 未见传感器 | 迁移评估 |
| TactileUMI | 真实机器人 | 未见传感器 | 迁移评估 |

**数据分布（重采样后）**: ~20% 人类数据、~30% 灵巧手机器人、~50% 夹爪机器人

### 实现细节

- **预训练 Backbone**: 视觉语言专家基于 [[π0.5]] 权重初始化
- **触觉专家**: 300M 参数共享 Transformer
- **评估**: 仿真 100 rollouts/任务，真实机器人 20 rollouts/任务
- **硬件**: 分布于 5 家合作机构

### 消融实验

**FTP-1 vs NTP-1（无触觉预训练）**:
- NTP-1：与 FTP-1 相同预训练数据和设置，但排除触觉输入和触觉架构，微调时再加入触觉分支
- 结果：在 UniVTAC（已见），NTP-1 优于 FTP-π₀.₅，说明数据量有正面效果
- 结论：在 FlexivXense（未见），FTP-1 比 NTP-1 高出 **+37.5%**，证明触觉专家的预训练是迁移的核心

---

## 批判性思考

### 优点

1. **系统性解决异构性问题**: MTTS 从形态学角度统一不同传感器，比简单 concatenation 或 adapter 方法更有语义基础
2. **验证充分**: 同时覆盖仿真（UniVTAC）和真实机器人，既有已见传感器又有未见传感器，消融实验明确区分预训练效果与数据量效果
3. **数据生态贡献**: 26 来源的数据整合本身是重要工程贡献，Sharpa North-FTP-1 数据集的新采集也丰富了社区资源

### 局限性

1. **缺乏基于触觉的伺服控制**: 当前框架主要关注触觉感知用于动作预测，没有探索触觉反馈闭环伺服（tactile servoing）和低级力控
2. **数据规模受限**: 3000 小时相较视觉 VLA 的数据规模仍属较小，未来需验证 scaling law 在触觉域是否成立
3. **触觉预测/世界模型未探索**: 未将触觉信号用于触觉预测（tactile prediction）或世界模型建模

### 潜在改进方向

1. 结合触觉的主动感知（active tactile perception）和伺服控制，实现力/位混合控制
2. 扩大数据规模并研究触觉策略的 scaling law
3. 探索触觉世界模型：预测接触状态变化，辅助规划与泛化

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码链接待确认）
- [ ] 预训练模型（待发布）
- [x] 训练细节（Appendix 中有部分描述）
- [x] 数据集描述（汇聚来源和新收集数据集有描述）

---

## 关联笔记

### 基于

- [[π0.5]]: 视觉语言动作专家初始化来源，FTP-1 在其架构基础上扩展触觉分支
- [[T3]]: 用于图像型触觉传感器的共享预训练 Transformer 编码器

### 对比

- [[VITaL]]: 触觉视觉-语言策略，绑定特定传感器配置
- [[UniVTAC-ACT]]: 仿真多任务触觉 ACT 策略，UniVTAC 基准的对比方法
- [[Tactile-VLA]]: 触觉增强 VLA，FTP-1 的直接竞争对比
- [[π0.5]]: 无触觉通用策略，验证加触觉后的增益

### 方法相关

- [[MTTS]]: 形态感知触觉 Token 空间（本文核心创新）
- [[流匹配]]: 动作专家所用的生成建模方法
- [[动作块]]: Action Chunk 输出格式
- [[RMSNorm]]: 用于本体感知自适应注入
- [[Fourier 编码]]: 用于状态型触觉信号的位置编码

### 硬件/数据相关

- [[GelSight]]: 图像型触觉传感器代表
- [[Sharpa North]]: 本文新增数据集使用的灵巧手平台

---

## 速查卡片

> [!summary] FTP-1: Generalist Foundation Tactile Policy
> - **核心**: 首个跨 21 种触觉传感器的通用基础触觉策略，支持图像/数组/状态三类输入
> - **方法**: MTTS（24 功能区 slot）统一 token 化 + 异构编码器 + 300M 共享触觉专家 + 多专家融合（基于 π₀.₅）
> - **结果**: 已见传感器 +17.2%、未见传感器 +31%（仿真+真机双验证）
> - **代码**: [ftp1-policy.github.io](https://ftp1-policy.github.io/)

---

*笔记创建时间: 2026-06-14*
