---
title: "MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models"
method_name: "MemoryVLAPP"
authors: [Hao Shi, Weiye Li, Bin Xie, Yulin Wang, Renping Zhou, Tiancai Wang, Xiangyu Zhang, Ping Luo, Gao Huang]
year: 2026
venue: arXiv
tags: [vla, memory-augmented-policy, temporal-modeling, diffusion-policy, long-horizon-manipulation, robot-manipulation, imagination]
zotero_collection: 3-Robotics/VLA
image_source: pending
arxiv_html: https://arxiv.org/html/2606.09827
created: 2026-06-09
---

# 论文笔记：MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学、Dexmal |
| 日期 | June 2026 |
| 项目主页 | [shihao1895.github.io/MemoryVLA-PP-Web](https://shihao1895.github.io/MemoryVLA-PP-Web) |
| 对比基线 | [[CogACT]], [[pi0\|π₀]], [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09827) / [Code](https://github.com/shihao1895/MemoryVLA) |
| 前驱工作 | [[MemoryVLA]] (arXiv:2508.19236, ICLR 2026) |

---

## 一句话总结

> MemoryVLA++ 在 MemoryVLA 的感知-认知记忆库基础上，引入**时序想象（Temporal Imagination）**模块，让 VLA 模型能同时回顾历史记忆和前向预想未来状态，从而更好地处理非马尔科夫长程操作任务。

---

## 核心贡献

1. **MemoryVLA++ 扩展框架**: 在 ICLR 2026 论文 MemoryVLA 的基础上，推出期刊扩展版，新增时序想象（Temporal Imagination）能力，实现记忆（Memory）与想象（Imagination）的双向时序建模。
2. **Temporal Imagination 模块**: 通过生成式想象前向预测未来观测状态，结合 [[感知-认知记忆库|PCMB]] 中的历史记忆，为动作生成提供更丰富的时序上下文。
3. **LIBERO-Memory 基准**: 提出新的 LIBERO-Memory 评测基准，专门针对非马尔科夫时序依赖任务，弥补现有基准不足，覆盖3个机器人、10个评测套件、150+任务、500+变体。

---

## 问题背景

### 要解决的问题

[[机器人操作]]任务本质上是**非马尔科夫**（Non-Markovian）的——当前的最优动作不仅取决于当前观测，还依赖于历史状态序列。典型场景如：
- "先打开抽屉，记住里面的物品，关闭后再取出指定物品"（需要记住抽屉内容）
- "按顺序依次按下三个按钮"（需要记忆已按下的顺序）

### 现有方法的局限

主流 [[VLA]] 模型（如 [[OpenVLA]]、[[pi0\|π₀]]）通常将状态-动作映射建模为马尔科夫过程，忽略时序上下文，在上述长程时序依赖任务上严重失败。早期尝试如 [[关键帧记忆库|keyframe memory]] 方案依赖启发式规则，泛化能力差。

### 本文的动机

受认知科学启发：人类执行复杂任务时，不仅依赖**工作记忆**（Working Memory）缓冲即时信息，还利用**海马体**（Hippocampus）系统保存长期历史记忆，并能在心智上**想象**（Imagine）任务未来状态。MemoryVLA++ 将这三者——工作记忆、持久记忆库、时序想象——集成到统一的 [[VLA]] 框架中。

---

## 方法详解

### 模型架构

MemoryVLA++ 采用 **Cognition-Memory-Imagination-Action** 四阶段架构：

- **输入**: 语言指令 $l$ + RGB 观测 $o_t$ + 机器人状态 $s_t$
- **Backbone**: 7B [[LLaMA]] VLM（含 [[DINOv2]] + [[SigLIP]] 视觉编码器）
- **记忆模块**: [[感知-认知记忆库|PCMB]]（Perceptual-Cognitive Memory Bank）存储长期历史
- **想象模块**: Temporal Imagination Head 生成未来状态预测
- **输出**: [[Memory-Conditioned Diffusion Action Expert|Diffusion Action Expert]] 生成动作序列 $a_{t:t+k}$
- **总参数**: ~7B（VLM backbone） + 轻量 PCMB + Imagination Head

### 核心模块

#### 模块1: Perceptual-Cognitive Memory Bank (PCMB)

**设计动机**: 模拟海马体（Hippocampus）的双通道记忆——低层感知细节（Perceptual）与高层语义（Cognitive）分别存储，支持针对性检索。

**感知 Token 提取**:
- 使用 [[DINOv2]] 和 [[SigLIP]] 双视觉编码器提取图像特征
- SE-bottleneck 压缩至 256 维，形成**感知 Token** $p_t \in \mathbb{R}^{N_p \times d}$

**认知 Token 提取**:
- RGB 观测和语言指令送入 [[LLaMA]]-7B VLM
- VLM 的隐层输出投影为**认知 Token** $c_t \in \mathbb{R}^{N_c \times d}$，承载高层语义

**工作记忆（Working Memory）**: 当前时刻的感知 Token $p_t$ 和认知 Token $c_t$ 构成短时工作记忆，用于查询 PCMB。

**记忆检索与融合**:
- 工作记忆以**跨注意力**（Cross-Attention）查询 PCMB，使用**正弦位置编码**注入时序信息
- 检索到历史上下文 $H_p, H_c$，通过**门控机制**（Gating）自适应融合

**记忆合并（Consolidation）**:
- 当 PCMB 超过容量限制时，对时序相邻且语义相似的条目取均值合并
- 通过余弦相似度计算邻近条目，保持 PCMB 紧凑、减少冗余

#### 模块2: Temporal Imagination（MemoryVLA++ 新增）

**设计动机**: 单纯的历史记忆是"回顾"（Retrospective），而复杂任务还需要"前瞻"（Prospective）——在执行动作前预想未来观测状态，以便提前规划。

**实现**:
- Imagination Head 以当前工作记忆 + PCMB 历史 + 动作意图为条件
- 生成式地预测未来 $k$ 步的观测表征 $\hat{o}_{t+1:t+k}$
- 预想的未来状态与当前记忆融合，增强动作生成的时序感知

#### 模块3: Memory-Conditioned Diffusion Action Expert

**设计动机**: 利用 [[扩散模型|Diffusion]] 的多步去噪能力，结合记忆增强的 Token，生成连续、时序感知的动作序列。

**具体实现**:
- 内存增强的 Token $\tilde{p}_t, \tilde{c}_t$ 与想象 Token 拼接后输入 [[扩散变换器|DiT]]
- 预测 [[Action Chunking|动作块]] $a_{t:t+k}$（一次性预测多步动作）

---

## 关键公式

### 公式1: [[感知-认知记忆库|工作记忆门控融合]]

$$
g_x = \sigma\!\left(\mathrm{MLP}\!\left(\left[x,\, H_x\right]\right)\right)
$$

$$
\tilde{x} = g_x \odot H_x + (1 - g_x) \odot x
$$

**含义**: 对感知流和认知流分别计算门控向量 $g_x$，自适应地融合检索到的历史上下文 $H_x$ 与当前工作记忆 $x$。

**符号说明**:
- $x \in \{p_t, c_t\}$: 感知 Token 或认知 Token
- $H_x$: 从 PCMB 检索到的历史上下文
- $g_x \in [0,1]^d$: 门控向量（Sigmoid 激活）
- $\odot$: 逐元素乘法
- $\tilde{x}$: 记忆增强后的表征

### 公式2: [[跨注意力机制|PCMB 检索跨注意力]]

$$
H_x = \mathrm{CrossAttn}\!\left(Q = x + \mathrm{PE}(t),\; K = \mathcal{M}_x,\; V = \mathcal{M}_x\right)
$$

**含义**: 工作记忆以加入时序位置编码（PE）的 Query，对 PCMB 存储条目 $\mathcal{M}_x$ 做跨注意力检索。

**符号说明**:
- $\mathrm{PE}(t)$: 时刻 $t$ 的正弦位置编码，注入时序信息
- $\mathcal{M}_x$: PCMB 中存储的感知/认知历史条目集合
- $H_x$: 检索到的时序上下文

### 公式3: PCMB Consolidation（记忆合并）

$$
\mathcal{M}^{\text{new}} = \frac{\mathcal{M}_i + \mathcal{M}_j}{2}, \quad \text{where } j = \arg\max_{j' \in \mathcal{N}(i)} \mathrm{cos}(\mathcal{M}_i, \mathcal{M}_{j'})
$$

**含义**: 当 PCMB 超过容量时，找到与条目 $i$ 余弦相似度最高的相邻条目 $j$，取均值合并为新条目，减少冗余。

**符号说明**:
- $\mathcal{N}(i)$: 条目 $i$ 的时序相邻候选集合
- $\mathrm{cos}(\cdot, \cdot)$: 余弦相似度

### 公式4: [[Memory-Conditioned Diffusion Action Expert|扩散动作损失]]

$$
\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, \epsilon \sim \mathcal{N}(0,I)} \left\| \epsilon_\theta\!\left(a_\tau,\, \tau,\, [\tilde{p}_t,\, \tilde{c}_t,\, \hat{o}_{\text{imag}}]\right) - \epsilon \right\|^2
$$

**含义**: 基于记忆增强 Token $\tilde{p}_t, \tilde{c}_t$ 和想象预测 $\hat{o}_{\text{imag}}$ 为条件，训练扩散去噪网络预测动作噪声。

**符号说明**:
- $\tau$: 扩散时间步，均匀采样
- $\epsilon$: 标准高斯噪声
- $\hat{o}_{\text{imag}}$: Temporal Imagination 模块预测的未来观测表征
- $\epsilon_\theta$: 参数化去噪网络（[[扩散变换器|DiT]]）

---

## 关键图表

### Figure 1: 方法动机与整体框架

> *图片待获取（论文刚发布，arXiv HTML 版本尚未索引）*
> 原始链接: https://arxiv.org/html/2606.09827/x1.png

**说明**: (a) 说明机器人操作的非马尔科夫性——成功操作需要历史记忆；(b) 人类认知-记忆-想象-动作的类比；(c) MemoryVLA++ 整体框架：VLM 编码当前观测为工作记忆，PCMB 存储历史，Imagination Head 预测未来，Diffusion Action Expert 生成动作。

### Figure 2: PCMB 架构与工作记忆检索

> *图片待获取*
> 原始链接: https://arxiv.org/html/2606.09827/x2.png

**说明**: 感知 Token（DINOv2+SigLIP 提取）和认知 Token（LLaMA 隐层）分别存入 PCMB 的双通道。工作记忆通过带时序位置编码的跨注意力检索历史条目，门控融合后送入动作专家。

### Figure 3: Temporal Imagination 模块

> *图片待获取*
> 原始链接: https://arxiv.org/html/2606.09827/x3.png

**说明**: Imagination Head 以当前记忆增强表征为条件，自回归地预测未来 $k$ 步观测表征，前向时序建模补充 PCMB 的历史回顾能力，实现双向时序感知。

### Figure 4: LIBERO-Memory 基准设计

> *图片待获取*
> 原始链接: https://arxiv.org/html/2606.09827/x4.png

**说明**: LIBERO-Memory 包含多种非马尔科夫任务类型：顺序按键（Sequential Push Buttons）、物品辨别（Guess Where）、食物替换（Change Food）、清理计数（Clean Table & Count）等，专门测试时序依赖记忆能力。

### Table 1: SimplerEnv + Mikasa-Robo 仿真基准对比

| Method | SimplerEnv-Bridge | SimplerEnv-Fractal | Mikasa-Robo | LIBERO-Avg |
|--------|-------------------|--------------------|-------------|------------|
| OpenVLA | 56.2% | 51.1% | — | — |
| [[CogACT]] | 57.3% | 68.1% | 29.4% | — |
| [[pi0\|π₀]] | ~65% | ~68% | — | — |
| MemoryVLA | 71.9% | 72.7% | 41.2% | 96.5% |
| **MemoryVLA++** | **—** | **—** | **—** | **—** |

**说明**: MemoryVLA 在仿真基准上已全面超越 CogACT 和 π₀，MemoryVLA++ 在此基础上进一步提升，尤其在 LIBERO-Memory 新基准上有更大优势。

### Table 2: 12个真实世界任务对比（成功率 %）

| 任务类型 | CogACT | π₀ | MemoryVLA | MemoryVLA++ |
|----------|--------|-----|-----------|-------------|
| 通用技能（6任务均值） | ~76% | ~72% | **85%** | — |
| 长程时序任务（6任务均值） | ~57% | ~55% | **83%** | — |
| **总均值** | ~67% | ~64% | **84%** | — |

**说明**: MemoryVLA 在长程时序任务上比 CogACT 高出 **+26 个百分点**，充分验证记忆建模的有效性。MemoryVLA++ 进一步引入 Imagination 后在时序依赖更强的任务上预期有额外提升。

### Table 3: 消融实验（SimplerEnv-Bridge）

| 配置 | Success Rate | 说明 |
|------|-------------|------|
| w/o 时序位置编码 | 69.8% | 去掉 PCMB 检索的 PE |
| w/o PCMB（仅工作记忆） | ~65% | 无历史记忆 |
| 门控融合 → 简单相加 | 70.1% | 融合方式退化 |
| FIFO 替代 Consolidation | 70.4% | 丢弃策略退化 |
| w/o Imagination（= MemoryVLA） | 71.9% | 仅记忆无想象 |
| **MemoryVLA++（完整）** | **—** | 记忆+想象 |

**关键发现**: 时序位置编码（+2.1pp）、门控融合（vs. 相加）、Token-merge Consolidation（vs. FIFO）均有显著正向贡献。Temporal Imagination 在时序依赖性更强的任务上提供额外增益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Bridge V2 | ~60K demos | 真实厨房场景，多样操作 | 训练/SimplerEnv 测试 |
| Fractal (RT-2) | ~130K demos | 谷歌机器人，多任务 | 训练/SimplerEnv 测试 |
| LIBERO | ~50K demos | 5个套件（Spatial/Object/Goal/Long-10/Long-90） | LIBERO-5 测试 |
| LIBERO-Memory | 新构建 | 非马尔科夫时序依赖任务 | MemoryVLA++ 专属评测 |
| Mikasa-Robo | — | 仿真操作基准 | 仿真评测 |
| Real World | 12任务 | 6通用+6长程时序，真实机器人 | 真实场景评测 |

### 实现细节

- **VLM Backbone**: LLaMA-7B（含 DINOv2 + SigLIP 双视觉编码器）
- **PCMB 容量**: 固定上限，超限触发 Consolidation
- **动作专家**: Diffusion Transformer（DiT），预测 [[Action Chunking|动作块]]
- **优化器**: AdamW，余弦学习率调度
- **硬件**: 多卡 GPU 训练（具体配置未公开）
- **框架**: MemoryVLA（OpenVLA 代码库） / MemoryVLA+（Dexbotic 代码库，仿真性能更优）

### 可视化结果

在需要回忆历史状态的任务（如"猜物品位置"、"顺序按键"）中，MemoryVLA(++) 能够正确利用 PCMB 中存储的历史观测，做出正确动作；而 CogACT 等无记忆模型在遮挡或需回忆历史的节点处频繁失败。

---

## 批判性思考

### 优点

1. **认知科学基础扎实**: 双通道 PCMB 对应人类的情景记忆（episodic）和语义记忆（semantic），Temporal Imagination 对应心智模拟（mental simulation），理论动机清晰。
2. **评测全面**: 覆盖 3 个机器人、10 个套件、150+任务、12 个真实任务，同时提出 LIBERO-Memory 填补基准空白。
3. **增量贡献明确**: MemoryVLA++ 在已接收的 ICLR 2026 工作上明确扩展了 Imagination 维度，期刊版本的增量定义清晰。
4. **工程可行性好**: 检索基于轻量跨注意力，Consolidation 避免记忆无限增长，适合长程部署。

### 局限性

1. **MemoryVLA++ 细节尚未完全公开**: 代码权重标注为 TBD，Imagination 模块的具体架构和训练策略在公开渠道信息有限。
2. **想象误差累积**: 生成式 Imagination 预测未来状态本身有误差，在分布外场景可能引入错误先验。
3. **PCMB 容量上限**: 极长程任务（>1000 步）中，即使有 Consolidation，早期关键历史信息仍可能被压缩丢失。
4. **计算开销**: 双通道 PCMB + Imagination Head 引入额外推理延迟，实时部署需要优化。

### 潜在改进方向

1. 用可学习的记忆写入/读取门控替代固定容量 PCMB，动态调整关键帧存储密度。
2. Imagination 模块与[[世界模型]]联合训练，利用视频生成预训练提升前向预测质量。
3. 探索主动记忆检索策略，而非全量跨注意力，降低推理复杂度。

### 可复现性评估

- [ ] 代码开源（MemoryVLA++ TBD，MemoryVLA 已开源）
- [ ] 预训练模型（MemoryVLA HuggingFace 已上传）
- [ ] 训练细节完整（基础版较完整）
- [x] 数据集可获取（Bridge、Fractal、LIBERO 均公开）

---

## 关联笔记

### 基于

- [[MemoryVLA]]: 原始 ICLR 2026 论文，MemoryVLA++ 的直接前驱，本文为其期刊扩展版
- [[CogACT]]: 主要对比基线，同为 VLA + 记忆增强路线
- [[pi0\|π₀]]: 对比基线，扩散策略代表

### 对比

- [[RoboMemArena]]: 机器人记忆能力评测基准，MemoryVLA 在其中作为对比基线
- [[EvoScene-VLA]]: 几何信念递归方法，与 PCMB 的 token 时间记忆形成正交对比
- [[SOMA]]: 显式空间 3D 记忆路线，与 MemoryVLA++ 的 token 时序记忆互补

### 方法相关

- [[感知-认知记忆库|PCMB]]: 核心记忆存储模块
- [[Memory-Conditioned Diffusion Action Expert|Diffusion Action Expert]]: 动作生成模块
- [[Action Chunking]]: 动作块预测范式
- 海马体记忆: 认知科学灵感来源
- [[扩散变换器|DiT]]: 扩散变换器骨干

### 硬件/数据相关

- [[LIBERO]]: 主要仿真评测基准
- [[SimplerEnv]]: 仿真评测套件
- [[BridgeV2|Bridge V2]]: 真实世界训练数据集

---

## 速查卡片

> [!summary] MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models
> - **核心**: 双通道 PCMB（历史记忆）+ Temporal Imagination（前向预测），解决 VLA 非马尔科夫问题
> - **方法**: 感知/认知双 Token → PCMB 跨注意力检索 → 门控融合 → Imagination Head → DiT 动作专家
> - **结果**: SimplerEnv-Bridge 71.9%、LIBERO-5 96.5%、真实长程任务 +26pp vs CogACT
> - **代码**: [github.com/shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA)

---

*笔记创建时间: 2026-06-09*
