---
title: "FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization"
method_name: "FlowPRO"
authors: [Yihao Wu, He Zhang, Junbo Tan, Xueqian Wang, Zhengyou Zhang]
year: 2026
venue: arXiv
tags: [vla-post-training, preference-optimization, flow-matching, reward-free, bimanual-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.05468
created: 2026-06-09
---

# 论文笔记：FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent Robotics X / Futian Laboratory / Tsinghua University |
| 日期 | June 2026 |
| 项目主页 | [flowpro](https://wuyeyexvnainai.github.io/flowpro/) |
| 对比基线 | [[DAgger]] / [[π₀]] / Flow-DPO / TPO |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05468) |

---

## 一句话总结

> FlowPRO 提出 RPRO 损失函数，通过近端正则化消除 Flow-DPO 的奖励黑客问题，并用人工干预-回滚范式在真实机器人上无奖励地离线收集偏好数据，实现 flow-matching VLA 的后训练。

---

## 核心贡献

1. **RPRO 损失函数**: 专为 [[Flow-DPO|Flow-matching DPO]] 设计的近端偏好优化目标，通过对比优化器 + 显式近端正则化器锚定隐式奖励的绝对量级，消除普通 Flow-DPO 的奖励黑客失效模式
2. **干预-回滚数据收集范式**: 单个操作员遥控动作同时产生自然配对的正/负轨迹，并通过 [[平滑插值|Smooth Interpolation]] 将稀疏的轨迹偏好监督转化为密集的逐状态偏好监督
3. **真实机器人验证**: 在 Dobot XTrainer 双臂平台上四个长程任务全面超越四类基线（DAgger、DAgger-Buffered、π₀.6*、TPO），取得最高成功率与最短完成时间

---

## 问题背景

### 要解决的问题

[[VLA 后训练]] 是将预训练 [[VLA]] 模型变成可靠真实机器人策略的主要瓶颈。部署预训练 VLA（如 [[π₀]]、[[π₀.₅]]）到特定任务时，需要高效的后训练机制提升任务成功率。

### 现有方法的局限

- **[[SFT]] / [[DAgger]]**: 仅间接利用失败信号（DAgger 只收集正轨迹，无显式偏好对比），数据效率低
- **奖励驱动 RL**: 真实环境中设计奖励函数困难，训练可靠评价器（critic）代价高昂
- **普通 [[Flow-DPO]]**: 隐式奖励无绝对量级约束，训练中发生奖励黑客（reward hacking）——策略学会最大化对比项而非真正改善任务表现

### 本文的动机

flow-matching 动作头的策略空间与语言模型不同，需要专门设计的偏好优化目标。通过近端正则化约束隐式奖励范围，可以在不引入显式奖励信号的情况下稳定后训练过程。

---

## 方法详解

### 整体框架

FlowPRO 由两个相互配合的部分组成：

- **数据侧**: 干预-回滚范式 + 平滑插值 → 生成密集偏好对
- **算法侧**: RPRO 损失 → 稳健地利用偏好对微调 [[Flow Matching|flow-matching]] VLA

### 核心模块

#### 模块1: 干预-回滚偏好数据收集

**设计动机**: 无需额外成本收集自然的正/负轨迹对，避免单独录制失败 rollout 的麻烦。

**具体实现**:

- 操作员遥控机器人执行任务，当策略表现良好时记录为**正轨迹**
- 当策略出现失误时，操作员**介入（intervention）**纠正，纠正完成后**回滚（rollback）**到介入前的状态重新由策略执行
- 单次操作员动作同时产生一对 $(τ^+, τ^-)$，且两者起始状态不同（per-pair 状态多样化）

**[[平滑插值|Smooth Interpolation]]（SI）**:

将轨迹级别的稀疏偏好监督扩展为逐状态的密集偏好监督：对正负轨迹在时间维度上进行状态插值，生成每个时间步的对应偏好对。

**批次混合（Batch Mixing）**:

固定比例的批次混合策略，上调当前轮次的数据权重，确保新收集数据的有效利用。

#### 模块2: RPRO（Robotic Flow-matching Proximalized Preference Optimization）

**设计动机**: 普通 [[Flow-DPO]] 仅有对比项，隐式奖励无上界约束，容易导致奖励黑客。

**具体实现**:

- **对比优化器**: 最大化正轨迹流匹配向量场与负轨迹流匹配向量场之间的对比差距
- **近端正则化器（Proximal Regularizer）**: 显式约束隐式奖励的绝对量级，锚定 $\pi_\theta$ 与参考策略 $\pi_{\text{ref}}$ 的距离，防止优化过冲

---

## 关键公式

### 公式1: [[Flow Matching|Flow-Matching 条件向量场]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

**含义**: flow-matching 基础训练目标，学习从噪声 $x_0$ 到干净动作 $x_1$ 的条件向量场 $v_\theta$

**符号说明**:
- $t \sim \mathcal{U}(0,1)$: 插值时间步
- $x_t = (1-t)x_0 + t x_1$: 线性插值中间状态
- $v_\theta$: 策略网络预测的向量场
- $x_0$: 噪声（标准正态采样）
- $x_1$: 干净动作

### 公式2: [[Flow-DPO|Flow-DPO 对比目标]]

$$
\mathcal{L}_{\text{Flow-DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \cdot \Delta r(τ^+, τ^-) \right) \right]
$$

其中隐式奖励差由正负轨迹的流匹配损失差给出：

$$
\Delta r(τ^+, τ^-) = \log \frac{\pi_\theta(τ^+)}{\pi_{\text{ref}}(τ^+)} - \log \frac{\pi_\theta(τ^-)}{\pi_{\text{ref}}(τ^-)}
$$

**含义**: 直接偏好优化在 flow-matching 动作空间的推广，利用正/负轨迹对进行对比学习

**符号说明**:
- $τ^+$: 正（preferred）轨迹
- $τ^-$: 负（rejected）轨迹
- $\pi_{\text{ref}}$: 冻结的参考策略（SFT checkpoint）
- $\beta$: 对比强度超参数
- $\sigma$: sigmoid 函数

### 公式3: [[RPRO|RPRO 近端偏好优化损失]]

$$
\mathcal{L}_{\text{RPRO}} = \mathcal{L}_{\text{Flow-DPO}} + \lambda \cdot \mathcal{L}_{\text{proximal}}
$$

其中近端正则化项：

$$
\mathcal{L}_{\text{proximal}} = \mathbb{E}_{τ^+} \left[ \left( \log \frac{\pi_\theta(τ^+)}{\pi_{\text{ref}}(τ^+)} \right)^2 \right] + \mathbb{E}_{τ^-} \left[ \left( \log \frac{\pi_\theta(τ^-)}{\pi_{\text{ref}}(τ^-)} \right)^2 \right]
$$

**含义**: 在对比损失基础上增加近端约束，锚定正负轨迹各自的隐式奖励绝对量级，防止奖励黑客

**符号说明**:
- $\lambda$: 近端正则化权重（超参数）
- $\mathcal{L}_{\text{proximal}}$: 对正/负轨迹分别约束的二次近端惩罚
- 其余符号与 Flow-DPO 一致

### 公式4: [[平滑插值|Smooth Interpolation 逐状态偏好扩展]]

对于干预时刻 $t^*$ 处的状态 $s^*$，以正轨迹状态 $s_t^+$ 和负轨迹状态 $s_t^-$ 插值：

$$
s_t^{\text{interp}} = (1 - \alpha_t) \cdot s_t^- + \alpha_t \cdot s_t^+, \quad \alpha_t = \frac{t}{t^*}
$$

**含义**: 将轨迹级别的稀疏偏好扩展为密集逐状态监督，提升数据利用率

**符号说明**:
- $t^*$: 操作员干预时刻
- $\alpha_t$: 随时间线性增大的插值系数
- $s_t^+$: 正轨迹在时刻 $t$ 的状态
- $s_t^-$: 负轨迹在时刻 $t$ 的状态

---

## 关键图表

### Figure 1: FlowPRO 整体框架概览

![Figure 1 - FlowPRO Overview](https://arxiv.org/html/2606.05468/x1.png)

**说明**: 左侧为干预-回滚数据收集流程，操作员遥控机器人时一次操作产生正/负轨迹对；右侧为 RPRO 损失训练流程，对比项 + 近端正则化项共同优化 flow-matching VLA。

### Figure 2: 干预-回滚数据收集范式

![Figure 2 - Intervention-and-Rollback](https://arxiv.org/html/2606.05468/x2.png)

**说明**: 展示从单次操作员干预动作如何同时生成正/负轨迹对，以及 Smooth Interpolation 如何将稀疏轨迹偏好扩展为密集逐状态监督信号。

### Figure 3: RPRO 与 Flow-DPO 对比——奖励黑客现象

![Figure 3 - Reward Hacking Analysis](https://arxiv.org/html/2606.05468/x3.png)

**说明**: 展示普通 Flow-DPO 在训练中出现奖励黑客（隐式奖励无界增大但任务性能不升反降），而 RPRO 近端正则化有效锚定隐式奖励量级，训练稳定。

### Figure 4: 四个双臂任务示意

![Figure 4 - Bimanual Tasks](https://arxiv.org/html/2606.05468/x4.png)

**说明**: 在 Dobot XTrainer 平台上的四个任务：
- **Pack**: 需要亚厘米精度的插入操作
- **Cap**: 需要双臂在空中协调配合
- **USB**: 需要亚毫米精度插入
- **Case**: 可变形物体的长程打包任务

### Figure 5: 各方法成功率对比

![Figure 5 - Success Rate Comparison](https://arxiv.org/html/2606.05468/x5.png)

**说明**: FlowPRO 在所有四个任务上均取得最高成功率，超越 DAgger、DAgger-Buffered、π₀.6*（advantage conditioning）和 TPO（trajectory-wise preference optimization）四类基线。

### Figure 6: 消融实验——各组件贡献

![Figure 6 - Ablation Study](https://arxiv.org/html/2606.05468/x6.png)

**说明**: 逐步移除 RPRO 各组件（近端正则化、Smooth Interpolation、批次混合）后的性能衰减，验证每个设计决策的有效性。

### Table 1: 主实验结果——各方法成功率与完成时间

| 方法 | Pack | Cap | USB | Case | 平均成功率 | 平均完成时间 |
|------|------|-----|-----|------|-----------|------------|
| SFT baseline | - | - | - | - | 基准 | 基准 |
| DAgger | 正偏好-only | 正偏好-only | 正偏好-only | 正偏好-only | < FlowPRO | > FlowPRO |
| DAgger-Buffered | 正偏好-only | 正偏好-only | 正偏好-only | 正偏好-only | < FlowPRO | > FlowPRO |
| π₀.6* (advantage) | 正+负 | 正+负 | 正+负 | 正+负 | FlowPRO - 2~7pp | > FlowPRO |
| TPO (traj-wise) | 正+负 | 正+负 | 正+负 | 正+负 | < FlowPRO | > FlowPRO |
| **FlowPRO (RPRO)** | **最高** | **最高** | **最高** | **最高** | **最高** | **最短** |

**关键发现**: FlowPRO 在所有任务上均一致性地取得最高成功率和最短完成时间；π₀.6* 是最强基线，FlowPRO 领先 2~7 个百分点。

### Table 2: 消融实验——RPRO 组件归因

| 配置 | 平均成功率 | 说明 |
|------|-----------|------|
| Flow-DPO（无近端项） | 显著下降（奖励黑客） | 训练不稳定 |
| RPRO w/o Smooth Interpolation | 下降 | 稀疏偏好监督利用率低 |
| RPRO w/o Batch Mixing | 下降 | 新数据利用率低 |
| **RPRO（完整）** | **最高** | - |

**关键发现**: 近端正则化是最关键组件（移除后奖励黑客导致性能崩溃）；Smooth Interpolation 和 Batch Mixing 各自带来稳定增益。

---

## 实验

### 硬件平台与任务

| 任务 | 难点 | 长程步骤 |
|------|------|---------|
| Pack | 亚厘米精度插入 | 中 |
| Cap | 双臂空中协调 | 中 |
| USB | 亚毫米精度插入 | 短 |
| Case | 可变形物体长程打包 | 长 |

**平台**: Dobot XTrainer 双臂机器人

### 基础策略

实验以两个 [[π₀]] 系列模型为 base policy（均为 flow-matching 动作头），从相同 SFT checkpoint 出发进行后训练，验证 FlowPRO 的通用性。

### 实现细节

- **后训练策略**: 所有比较方法使用相同的迭代数据收集协议
- **基础策略**: π₀ 系列（具体为 π₀.6*的 base，原文表述为 "two π₀-family base policies"）
- **评估指标**: 任务成功率（%）+ 完成时间（秒）

---

## 批判性思考

### 优点

1. **无需奖励设计**: 完全绕开真实环境奖励函数的设计难题，操作员干预自然产生偏好标签
2. **理论严密**: 近端正则化有清晰的动机（锚定隐式奖励绝对量级），并通过实验验证了其消除奖励黑客的效果
3. **数据高效**: 单次操作员动作同时产生正/负对，Smooth Interpolation 进一步放大数据密度

### 局限性

1. **依赖人工干预**: 数据收集仍需操作员持续参与，规模化成本较高；与完全自主的在线 RL 相比灵活性有限
2. **仅验证双臂平台**: 所有实验在 Dobot XTrainer 上进行，对单臂、移动操作、足式等场景的泛化性未知
3. **数学形式推测**: 近端正则化的具体形式（对正/负轨迹分别施加 L2 约束）在论文中详细程度待验证，本笔记的公式基于合理推断

### 潜在改进方向

1. 结合自动成功/失败检测减少操作员依赖
2. 探索在仿真中预训练偏好对、真实环境验证的混合范式
3. 将 RPRO 推广到 autoregressive 动作头（如 OpenVLA 的 token 生成）

### 可复现性评估

- [ ] 代码开源（论文提及 project website，但未确认代码是否公开）
- [ ] 预训练模型（未知）
- [x] 训练细节基本完整（干预协议、损失权重等有描述）
- [ ] 数据集可获取（真实机器人数据，暂未公开）

---

## 关联笔记

### 基于

- [[π₀]]: base policy，flow-matching VLA 基础，FlowPRO 在其上进行后训练
- [[DPO]]: 直接偏好优化，RPRO 是其在 flow-matching 空间的近端化推广
- [[Flow-DPO]]: FlowPRO 的直接前驱，RPRO 通过近端正则化修复了其奖励黑客问题

### 对比

- [[DAgger]]: 正偏好-only 基线，FlowPRO 通过引入负偏好信号超越它
- [[π₀.₅]]: π₀ 系列模型，FlowPRO 的后训练目标 base policy 之一
- [[PAPO-VLA]]: 同类 VLA 偏好优化后训练工作

### 方法相关

- [[VLA 后训练]]: FlowPRO 所属的后训练范式
- [[Flow Matching]]: FlowPRO 目标模型的动作头架构
- [[RLHF]]: 更通用的人类反馈强化学习框架，FlowPRO 是其无奖励离线变体

### 硬件/数据相关

- [[Bimanual 任务族]]: 实验平台对应的双臂操作任务类型

---

## 速查卡片

> [!summary] FlowPRO (arXiv 2606.05468)
> - **核心**: 无奖励离线偏好优化框架，专为 flow-matching VLA 设计后训练
> - **方法**: RPRO 损失（对比项 + 近端正则化）+ 干预-回滚数据收集 + Smooth Interpolation
> - **结果**: Dobot XTrainer 双臂平台 4 任务全面超越 DAgger/TPO/π₀.6* 等 4 类基线
> - **代码**: [arXiv](https://arxiv.org/abs/2606.05468)（代码链接待确认）

---

*笔记创建时间: 2026-06-09*
