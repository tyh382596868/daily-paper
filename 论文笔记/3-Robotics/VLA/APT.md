---
title: "APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies"
method_name: "APT"
authors: [Kechun Xu, Zhenjie Zhu, Anzhe Chen, Rong Xiong, Yue Wang]
year: 2026
venue: arXiv
tags: [vla, action-expert, instruction-generalization, diffusion-policy, pretraining, bayesian-factorization, ood-generalization]
zotero_collection: 3-Robotics/VLA
image_source: pending
arxiv_html: https://arxiv.org/html/2606.12366
created: 2026-06-13
---

# 论文笔记：APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University |
| 日期 | June 2026 |
| 项目主页 | [xukechun.github.io/papers/APT](https://xukechun.github.io/papers/APT/) |
| 对比基线 | [[π₀.₅]], [[OpenVLA]], [[π₀]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12366) / Code: N/A |

---

## 一句话总结

> APT 通过两阶段训练先在视觉-动作对上预训练行动专家、再通过门控融合注入语言，从贝叶斯角度解决 VLA 策略在分布外语言指令上的泛化失败问题。

---

## 核心贡献

1. **贝叶斯分解视角**: 将策略 $\pi(a|v,\ell)$ 分解为视觉-动作先验 $\pi^p(a|v)$ 与语言似然 $\mathcal{L}(\ell|v,a)$ 的乘积，为[[Action Expert|行动专家]]预训练提供理论依据
2. **两阶段训练框架**: Stage 1 在冻结 VLM 上仅用视觉-动作对预训练行动专家，消除语言不平衡导致的捷径学习；Stage 2 通过门控融合注入语言对齐任务指令
3. **层级门控融合架构**: 从 [[Qwen3-VL]] 均匀采样 N 个中间特征，通过可学习标量门逐层注入行动专家，兼顾浅层空间与深层语义特征，跨主流 VLA 架构（π 风格 / GR00T 风格）通用

---

## 问题背景

### 要解决的问题

[[VLA]] 策略在测试时面对**分布外（OOD）语言指令**时严重失效——即便微调后语言格式一致，模型仍无法泛化到未见过的对象、容器或环境描述。

### 现有方法的局限

VLA 数据集存在结构性不平衡：一条轨迹中 $T$ 步视觉-动作对共享同一条语言指令，视觉-动作多样性至少是语言多样性的 $T$ 倍。这导致三级联式失败：

1. [[Action Shortcut|动作捷径学习]]：行动专家优先从视觉捷径预测动作，忽略语言
2. **梯度噪声**：捷径梯度反向传播污染 VLM，破坏语言表示
3. **OOD 泛化失败**：推理时对未见指令无法正确响应

具体方法的局限：
- **[[OpenVLA]]**（离散动作）：通过视觉-语言共训练缓解不平衡，但连续灵巧操作能力弱
- **[[π₀]]**（连续动作）：行动专家从随机初始化开始，无预训练保护
- **[[π₀.₅]]**（[[Knowledge Insulation|知识隔离]] KI）：停止行动专家→VLM 梯度，保护分布内语言，但 OOD 任务仍失败（单独 KI 无法突破瓶颈，见 Table 2）
- **LangForce**：最大化动作-语言互信息，获得 OOD 语言理解但损失视觉适应性

### 本文的动机

通过[[贝叶斯分解]]将语言条件策略解耦：先在纯视觉-动作对上训练稳健先验（避开语言不平衡），再以语言似然对其进行对齐。这样行动专家天然不会学习语言捷径，VLM 语言表示也得到保护。

---

## 方法详解

### 模型架构

APT 采用**两阶段预训练 + 任务微调**框架：

- **输入**: 语言指令 $\ell$ + 视觉观测 $v$ + 历史动作 $a_\text{hist}$ + 本体感知状态 $s_\text{prop}$ + 噪声动作 $a_\text{noisy}$
- **Backbone**: 冻结的 [[Qwen3-VL]]（提供逐层中间特征）
- **核心模块**: [[Action Expert|行动专家]]（扩散 Transformer）+ [[层级门控融合|层级门控融合机制]]
- **输出**: 连续动作序列 $a$
- **时间步注入**: [[FiLM|Feature-wise Linear Modulation (FiLM)]] 注入扩散时间步

### 核心模块

#### 模块 1：层级门控融合（Layer-wise Gated Fusion）

**设计动机**: 利用 [[Qwen3-VL]] 的多层中间表示，将浅层空间特征与深层语义特征同时注入行动专家，同时用可学习门控制各层注入强度，避免语言特征过度干扰视觉-动作先验。

**具体实现**:
- 从 Qwen3-VL 的等间隔深度位置均匀采样 $N$ 个中间层特征 $\phi_i^{\text{Qwen3-VL}}(v,\ell)$
- 每层行动专家输入添加门控残差：见 Eq. 3
- 可学习标量 $\hat{w}_i$（初始化为负值，使门初始接近零）通过 sigmoid 激活，实现渐进式注入
- 第 0 层直接使用 VLM 输入嵌入：$h_\text{in}^{(0)} = \phi_0^{\text{Qwen3-VL}}(v,\ell)$

#### 模块 2：两阶段训练

**Stage 1 — VA 先验预训练**:
- 仅激活行动专家的 $N/2$ 个注意力层
- 完全遮蔽来自 VLM 的语言 token，网络退化为纯视觉-动作函数：

$$[h_v,\, h_a] = \mathrm{SelfAttn}([v,\, a])$$

- 在大规模视觉-动作数据集上训练（Open X-Embodiment、DROID、RH20T、RoboMind 等）
- 结果：行动专家建立与语言无关的稳健视觉-动作先验，无捷径

**Stage 2 — VLA 似然对齐**:
- 扩展至 $N$ 个注意力层（原 $N/2$ 层后插入 $N/2$ 交错新层）
- 移除语言遮蔽，全 token 参与：

$$[h_v,\, h_\ell,\, h_a] = \mathrm{SelfAttn}([v,\, \ell,\, a])$$

- 原始 $N/2$ 层从 Stage 1 checkpoint 初始化，新层随机初始化
- 开启门控融合，联合训练所有 $N$ 层

**后训练**: 在目标任务数据上微调，可选择是否微调 VLM（Ft VLM）

#### 模块 3：行动专家多模态自注意力

动作 token 序列 $[a_\text{hist}, s_\text{prop}, a_\text{noisy}]$ 与视觉 token $v$、语言 token $\ell$ 通过分块因果自注意力（block-wise causal self-attention）共同处理。

---

## 关键公式

### 公式 1：[[贝叶斯分解|标准 VLA 目标]]

$$
\min -\mathbb{E}_{(a,v,\ell)\sim\mathcal{D}_\text{VLA}}\left[\log \pi(a|v,\ell)\right]
$$

**含义**: 标准 VLA 训练目标——最大化给定视觉观测 $v$ 和语言指令 $\ell$ 时动作 $a$ 的对数似然。

**符号说明**:
- $\mathcal{D}_\text{VLA}$：VLA 训练数据集
- $\pi(a|v,\ell)$：VLA 策略
- $(a, v, \ell)$：动作、视觉观测、语言指令三元组

---

### 公式 2：[[贝叶斯分解|APT 贝叶斯分解]]

$$
\pi(a|v,\ell) \propto \pi^p(a|v) \cdot \mathcal{L}(\ell|v,a)
$$

**含义**: 将策略按贝叶斯规则分解为视觉-动作先验与语言似然的乘积，使得先验可独立于语言进行预训练。

**符号说明**:
- $\pi^p(a|v)$：视觉-动作先验（VA prior），仅依赖视觉，不含语言
- $\mathcal{L}(\ell|v,a)$：语言条件似然（VLA likelihood），衡量动作与语言的对齐程度
- $\propto$：正比于（省略归一化常数）

---

### 公式 3：[[层级门控融合|门控融合更新规则]]

$$
h_\text{in}^{(i+1)} = h_\text{out}^{(i)} + \sigma(\hat{w}_i) \cdot \phi_i^{\text{Qwen3-VL}}(v,\ell)
$$

**含义**: 行动专家第 $i$ 层输出残差叠加门控后的 VLM 中间特征，作为第 $i+1$ 层输入，实现逐层语言-视觉语义注入。

**符号说明**:
- $h_\text{in}^{(i+1)}$：行动专家第 $i+1$ 层的输入隐状态
- $h_\text{out}^{(i)}$：行动专家第 $i$ 层的输出隐状态
- $\hat{w}_i$：第 $i$ 层可学习标量门控参数
- $\sigma(\cdot)$：Sigmoid 激活函数，将门控值压缩到 $(0,1)$
- $\phi_i^{\text{Qwen3-VL}}(v,\ell)$：从 Qwen3-VL 第 $i$ 个等间隔深度位置提取的中间特征

---

## 关键图表

### Figure 1：APT 动机与概览

> 🖼️ **Figure 1: Action expert pretraining enables effective instruction following** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.12366)）

**说明**: 说明行动专家预训练（APT）如何实现有效指令遵循。现有方法（如 [[π₀.₅]]）因数据不平衡导致 OOD 指令失败，APT 通过两阶段训练建立稳健的语言泛化能力。

---

### Figure 2：APT 两阶段训练架构

> 🖼️ **Figure 2: Overview of APT** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.12366)）

**说明**: Stage 1 中，[[Action Expert|行动专家]]仅条件化于来自冻结 [[Qwen3-VL]] 的视觉 token，构建 VA 先验；Stage 2 中，语言 token 通过[[层级门控融合]]注入，网络扩展至全 $N$ 层以对齐任务指令。

---

### Figure 3：行动专家设计与门控融合

> 🖼️ **Figure 3: Action Expert Design** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.12366)）

**说明**: 展示[[层级门控融合]]的具体结构。VLM 特征经 sigmoid 门控后以残差方式注入行动专家各层，动作 token 经分块因果自注意力处理，[[FiLM]] 模块注入扩散时间步。

---

### Figure 4：跨架构泛化消融

> 🖼️ **Figure 4: Action expert pretraining applies to diverse architectures** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.12366)）

**说明**: 验证两阶段预训练在 π 风格（APT）和 GR00T 风格（[[GR00T N1.5|GR00T]]）架构上均带来一致提升，APT 架构提升最为显著，证明方法的架构无关性。

---

### Figure 5：预训练必要性与语言注入机制消融

![Figure 5](https://arxiv.org/html/2606.12366/2606.12366v1/x5.png)

**说明**: 左图显示无大规模预训练的两阶段训练在未见类别（UO、UOUE）上显著落后；右图显示[[层级门控融合]]在所有维度上优于 Token Insertion，尤其在 VA 先验保持关键的 UO/UOUE 维度上差异最大。

---

### Figure 6：组合任务结果

![Figure 6](https://arxiv.org/html/2606.12366/2606.12366v1/x6.png)

**说明**: 组合任务中，[[π₀.₅]] 在任务链接（chaining）场景下成功率接近 0%，而 APT 维持较强性能，无需显式任务分割即可执行多指令链式操作。

---

### Figure 7：真实世界案例

![Figure 7](https://arxiv.org/html/2606.12366/2606.12366v1/x7.png)

**说明**: (a) 抓放任务、(b) 杂乱场景抓放、(c) 组合任务链接的真实机器人执行结果，展示 APT 在未见对象、容器、环境下的泛化能力及失败案例分析。

---

### Table 1：LIBERO-PRO 基准（成功率 %）

LIBERO-PRO 通过两类扰动评估 OOD 语言泛化：**Pos**（交换对象位置测试语言依赖度）和 **Task**（指令中替换对象测试 OOD 语言泛化）。

| 方法 | Spatial-Pos | Spatial-Task | Object-Pos | Object-Task | Goal-Pos | Goal-Task | Long-Pos | Long-Task | 平均 |
|------|------------|-------------|-----------|------------|---------|----------|---------|----------|------|
| OpenVLA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| π₀ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| π₀.₅ | 20 | 1 | 17 | 1 | 38 | 0 | 8 | 1 | 11 |
| LangForce | 11 | 48 | 10 | 10 | 4 | 11 | 2 | 15 | 14 |
| CaP-X | 12 | 14 | 22 | 18 | 26 | 17 | — | — | — |
| APT | 44 | 48 | 7 | 10 | 23 | 11 | 6 | 3 | 19 |
| **APT (Ft VLM)** | **62** | **62** | **24** | **17** | **10** | **20** | **12** | **12** | **27** |

**关键发现**: [[π₀.₅]] 和 [[π₀]] 在所有 Task 扰动维度几乎为 0，说明 KI 单独无法解决 OOD 语言问题；APT (Ft VLM) 在平均成功率上超越所有基线，尤其在 Spatial-Task 上达到 62%。

---

### Table 2：Pick-Place 消融基准（成功率 %）

| 方法 | KI | 2-Stage | Ft VLM | SO（同目标） | UO（未见对象） | UC（未见容器） | UOUE（未见对象+环境） |
|------|:--:|:-------:|:------:|:----------:|:------------:|:------------:|:------------------:|
| π₀ | ✓ | — | — | 42 | 30 | 26 | 16 |
| π₀.₅ | ✓ | ✓ | — | 84 | 70 | 86 | 50 |
| APT | ✓ | — | — | 88 | 56 | 66 | 34 |
| APT | — | ✓ | — | 90 | 58 | 40 | 40 |
| APT | — | ✓ | ✓ | 96 | 74 | 90 | 62 |
| **APT** | **✓** | **✓** | **✓** | **98** | **84** | **92** | **58** |

**关键发现**: 单独 KI 或单独两阶段训练提升有限，三者结合（KI + 2-Stage + Ft VLM）在所有 OOD 维度取得最佳成绩，UOUE 最难场景从 16% 提升至 58%。

---

### Table 3：真实世界实验（成功 / 总试验次数）

**Pick-Place 任务**:

| 方法 | SO（30次） | UO（20次） | UOUC（20次） | UOUCUE（40次） |
|------|-----------|-----------|------------|--------------|
| π₀.₅ | 27/30 | 11/20 | 9/20 | 16/40 |
| **APT** | **29/30** | **17/20** | **16/20** | **28/40** |

**Clutter Pick-Place 任务**:

| 方法 | SO（30次） | UC（30次） | UO（10次） | UOUE（10次） |
|------|-----------|-----------|-----------|------------|
| π₀.₅ | 18/30 | 18/30 | 4/10 | 3/10 |
| **APT** | **25/30** | **22/30** | **7/10** | **6/10** |

**关键发现**: APT 在真实机器人所有 OOD 设置下均优于 [[π₀.₅]]，杂乱场景下提升更为显著（UOUE: 3/10 → 6/10）。

---

## 实验

### 数据集与基准

| 数据集/基准 | 规模/特点 | 用途 |
|------------|---------|------|
| Open X-Embodiment | 大规模机器人数据集合 | Stage 1/2 预训练 |
| DROID | 大规模真实机器人操作数据 | 预训练 |
| RH20T | 大规模机器人操作数据 | 预训练 |
| RoboMind | 大规模机器人数据 | 预训练 |
| LIBERO-PRO | 基于 LIBERO 的 OOD 语言扰动基准 | 仿真测试 |
| Pick-Place Benchmark | SO/UO/UC/UOUE 四级 OOD 设置 | 消融 + 真实世界测试 |

### 实现细节

- **VLM Backbone**: Qwen3-VL（冻结用于 Stage 1，可选微调用于 Stage 2+）
- **行动专家**: 基于扩散的 Transformer，分块因果自注意力
- **时间步注入**: FiLM 模块
- **训练数据**: 多个大规模机器人数据集（Open X-Embodiment、DROID、RH20T、RoboMind）
- **完整实现细节**: 见论文附录 A

### 可视化结果

组合任务链接实验（Figure 6）显示 APT 可无缝执行跨任务链式指令，而 [[π₀.₅]] 在此场景完全失败（~0%），体现 APT 语言泛化能力在复杂指令组合上的突出优势。

---

## 批判性思考

### 优点

1. **理论基础扎实**: 贝叶斯分解为预训练阶段提供清晰动机，不是启发式设计
2. **架构无关性强**: 在 π 风格和 GR00T 风格架构上均验证有效，通用性好
3. **系统消融充分**: KI、两阶段训练、Ft VLM 三个组件逐一消融，贡献清晰

### 局限性

1. **无显式长时序记忆**: 论文自认缺乏长时序记忆建模，多步骤任务泛化能力受限
2. **评估场景局限**: 仅在桌面操作任务验证，未扩展到足式运动或移动操作
3. **数据依赖大规模预训练**: Figure 5 消融表明无大规模预训练时 OOD 性能显著下降，数据需求较高

### 潜在改进方向

1. 引入显式任务记忆模块以支持更长链的组合任务
2. 扩展到移动操作和足式运动等更多形态
3. 探索更少预训练数据下的高效两阶段方案

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节（附录 A 提供完整细节）
- [x] 数据集可获取（Open X-Embodiment 等公开数据集）

---

## 关联笔记

### 基于

- [[π₀]]: APT 的基础 VLA 架构，行动专家扩散策略框架来源
- [[π₀.₅]]: APT 的直接对比基线，Knowledge Insulation 方法
- [[Qwen3-VL]]: APT 使用的 VLM 骨干网络

### 对比

- [[π₀.₅]]: 同样使用知识隔离，但无两阶段预训练，OOD 指令泛化弱
- [[OpenVLA]]: 离散动作范式，视觉-语言共训练，连续操作能力有限
- [[GR00T N1.5]]: 消融研究中用于验证方法的 GR00T 风格架构

### 方法相关

- [[Action Expert|行动专家]]: APT 的核心组件
- [[层级门控融合]]: APT 的语言注入机制
- [[贝叶斯分解]]: APT 的理论基础
- [[Knowledge Insulation]]: APT 中 KI 组件的来源（来自 π₀.₅）
- [[FiLM]]: 行动专家中的时间步注入方式
- [[Diffusion Policy|扩散策略]]: 行动专家的核心生成框架

### 硬件/数据相关

- [[LIBERO]]: LIBERO-PRO 基准的底层框架

---

## 速查卡片

> [!summary] APT
> - **核心**: 贝叶斯分解 + 两阶段训练，解决 VLA 的 OOD 语言指令泛化问题
> - **方法**: Stage 1 纯视觉-动作预训练 + Stage 2 门控融合语言注入
> - **结果**: LIBERO-PRO 平均 27%（vs π₀.₅ 的 11%），真实 Pick-Place UOUCUE 28/40（vs 16/40）
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-06-13*
