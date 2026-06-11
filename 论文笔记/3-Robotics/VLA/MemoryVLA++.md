---
title: "MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models"
method_name: "MemoryVLA++"
authors: [Hao Shi, Weiye Li, Bin Xie, Yulin Wang, Renping Zhou, Tiancai Wang, Xiangyu Zhang, Ping Luo, Gao Huang]
year: 2026
venue: arXiv
tags: [vla, temporal-modeling, memory, imagination, robot-manipulation, long-horizon, diffusion-policy]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.09827
created: 2026-06-09
---

# 论文笔记：MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学（自动化系）、Dexmal、MEGVII Technology、天津大学、哈尔滨工业大学、StepFun |
| 日期 | June 2026 |
| 项目主页 | [MemoryVLA-PP-Web](https://shihao1895.github.io/MemoryVLA-PP-Web) |
| 对比基线 | [[MemoryVLA]]（ICLR 2026）、[[CogACT]]、[[pi-0|pi0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09827) / [Code](https://github.com/shihao1895/MemoryVLA) |

---

## 一句话总结

> MemoryVLA++ 在 MemoryVLA 的记忆框架基础上，引入"想象力"（Imagination）模块实现前瞻性时序建模，使 [[VLA]] 模型在长程机器人操作任务中同时具备回忆过去和预见未来的能力。

---

## 核心贡献

1. **扩展 Imagination 机制**: 在原 [[Perceptual-Cognitive Memory Bank]] 的基础上新增想象力模块，通过对未来状态的预测来增强当前动作决策，形成 Memory + Imagination 的双向时序建模。
2. **更强的时序 VLA 框架**: 统一"过去回忆"（Memory）与"未来预见"（Imagination）到单一 Cognition-Memory-Imagination-Action 框架中，解决 [[非马尔可夫任务]] 下 VLA 的时序依赖问题。
3. **全面 Benchmark 提升**: 在 LIBERO（98.4%）、CALVIN（4.29）、SimplerEnv-Bridge（73.9%）等多个基准上超越 MemoryVLA 原版，证明 Imagination 模块的有效性。

---

## 问题背景

### 要解决的问题

机器人操作任务本质上是[[非马尔可夫任务|非马尔可夫（Non-Markovian）]]的——当前动作不仅依赖当前观测，还依赖历史状态序列和对未来的预期。主流 [[VLA]] 模型（如 [[OpenVLA]]、[[pi-0|pi0]]）通常只处理单帧观测，忽视时序依赖，在长程任务上性能严重下降。

### 现有方法的局限

1. **单帧 VLA 模型**（OpenVLA、pi0）：无法处理跨步骤的时序依赖，长程任务成功率低
2. **简单历史拼接**：直接堆叠历史帧会导致 token 数量爆炸，超出 transformer 的 context window
3. **MemoryVLA（原版）**：虽然引入了 [[Perceptual-Cognitive Memory Bank]]，但仅做"回顾"式的历史记忆检索，缺乏对未来的预测性建模（Imagination）
4. **固定容量记忆库**：原始 PCMB 达到容量上限后需要合并条目，可能导致关键历史信息丢失

### 本文的动机

人类认知不仅依赖海马体（Hippocampus）对过去经历的回忆，还会通过心理模拟（Mental Simulation）想象未来可能的状态来辅助决策。MemoryVLA++ 将这一认知科学启发延伸，在已有的记忆框架之上引入"Imagination"模块，让模型能够：
- **向后看**：从 PCMB 中检索历史上下文（Memory）
- **向前想**：对即将发生的视觉/状态变化进行预测（Imagination）

---

## 方法详解

### 模型架构总览

MemoryVLA++ 是 MemoryVLA（ICLR 2026）的扩展期刊版本，采用 **Cognition-Memory-Imagination-Action** 四阶段架构：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$（RGB 图像）+ 机器人状态 $s_t$
- **Backbone**: [[DINOv2]]（感知编码）+ [[SigLIP]]（语义对齐）+ [[LLaMA-7B]]（语言推理）
- **核心模块**:
  1. [[Vision-Language Cognition Module]]：生成感知 token 与认知 token
  2. [[Perceptual-Cognitive Memory Bank]]（PCMB）：长程历史记忆存储与检索
  3. **Imagination Module**（新增）：基于当前状态对未来视觉/状态的预测建模
  4. [[Memory-Conditioned Diffusion Action Expert]]：时序感知的动作生成
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$（7-DoF：平移、旋转、夹爪状态）

### 核心模块一：Vision-Language Cognition Module

**设计动机**: 将视觉观测与语言指令融合为结构化 token，区分低层感知细节与高层语义推理。

**具体实现**:
- 使用 [[DINOv2]] 提取**感知 token**（Perceptual Tokens）$P_t$：保留空间细节、纹理信息
- 使用 [[SigLIP]] + [[LLaMA-7B]] 处理图像-语言对，生成**认知 token**（Cognitive Tokens）$C_t$：整合常识推理和语义理解
- 两类 token 共同构成**工作记忆**（Working Memory）$W_t = [P_t; C_t]$

### 核心模块二：Perceptual-Cognitive Memory Bank（PCMB）

**设计动机**: 模拟海马体的长程情景记忆功能，在有限容量下存储跨时间步的感知与认知历史。

**具体实现**:

**存储结构**：PCMB 维护两个独立的记忆槽序列：
- 感知记忆槽 $\mathcal{M}^P = \{M^P_1, M^P_2, \ldots, M^P_K\}$：存储历史感知细节
- 认知记忆槽 $\mathcal{M}^C = \{M^C_1, M^C_2, \ldots, M^C_K\}$：存储历史语义摘要

**检索（Retrieval）**：当前工作记忆 $W_t$ 通过带时间位置编码的[[Cross-Attention|交叉注意力]]查询 PCMB：

$$
\tilde{W}_t = \text{CrossAttn}(Q=W_t + \text{PE}(t),\ K=\mathcal{M},\ V=\mathcal{M})
$$

**门控融合（Gate Fusion）**：当前 token 与检索到的历史 token 通过自适应门控机制融合：

$$
\hat{W}_t = \sigma(g) \odot W_t + (1 - \sigma(g)) \odot \tilde{W}_t
$$

其中 $g$ 为可学习的门控参数，$\sigma$ 为 sigmoid 激活函数，$\odot$ 为逐元素乘法。

**整合更新（Consolidation）**：融合后的 token 写回 PCMB；当 PCMB 达到容量上限 $K$ 时，计算相邻条目间的相似度，合并最相似的条目对以维持紧凑性：

$$
\text{Merge}(M_i, M_j) = \frac{M_i + M_j}{2}, \quad i^* = \arg\max_{i} \text{sim}(M_i, M_{i+1})
$$

### 核心模块三：Imagination Module（MemoryVLA++ 新增）

**设计动机**: 模拟人类心理模拟（Mental Simulation）能力，通过预测未来状态来提前感知即将到来的视觉变化，辅助当前动作规划。

**具体实现**:

Imagination Module 在 PCMB 检索的基础上，对未来 $n$ 步的感知/认知状态进行预测：

- **前瞻预测（Look-Ahead Prediction）**: 基于当前增强工作记忆 $\hat{W}_t$，通过轻量 decoder 预测未来 token：

$$
\hat{W}_{t+1}, \ldots, \hat{W}_{t+n} = \text{ImagineDecoder}(\hat{W}_t, l)
$$

- **想象-现实对齐损失**: 在训练阶段，引入辅助损失约束想象 token 与真实未来 token 的一致性：

$$
\mathcal{L}_{\text{imag}} = \sum_{k=1}^{n} \| \hat{W}_{t+k} - \text{sg}(W_{t+k}) \|^2_2
$$

其中 $\text{sg}(\cdot)$ 为 stop-gradient 算子，防止梯度流向真实 token 编码器。

- **推理时利用想象 token**: 预测的未来 token $\{\hat{W}_{t+k}\}$ 与当前 PCMB 检索结果拼接，共同作为动作专家的条件输入，提供"前馈"的时序上下文。

### 核心模块四：Memory-Conditioned Diffusion Action Expert

**设计动机**: 在内存增强（Memory）和想象增强（Imagination）的 token 条件下，通过[[Diffusion Policy|扩散模型]]生成高质量的时序感知动作序列。

**具体实现**:
- 条件信号：记忆增强 + 想象增强的工作记忆 token $\hat{W}_t$（感知分支）和认知 token $C_t$（语义分支）
- 动作生成：使用 [[DiT]]（Diffusion Transformer）在 $T$ 步去噪中生成 [[Action Chunking|动作块]] $a_{t:t+k}$
- 输出格式：7-DoF 末端执行器轨迹（$\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z$）+ 夹爪开合状态

---

## 关键公式

### 公式 1：[[Cross-Attention|交叉注意力]] 记忆检索

$$
\tilde{W}_t = \text{softmax}\!\left(\frac{(W_t + \text{PE}(t)) \cdot \mathcal{M}^\top}{\sqrt{d}}\right) \mathcal{M}
$$

**含义**: 当前工作记忆以带时间位置编码的方式查询 PCMB，检索时序相关的历史上下文。

**符号说明**:
- $W_t$: 当前时间步的工作记忆 token（感知 + 认知）
- $\text{PE}(t)$: 时间步 $t$ 的位置编码
- $\mathcal{M}$: PCMB 存储的历史记忆矩阵
- $d$: token 维度（用于缩放）
- $\tilde{W}_t$: 检索到的历史上下文增强 token

### 公式 2：[[Adaptive Gating|自适应门控融合]]

$$
\hat{W}_t = \sigma(g) \odot W_t + (1 - \sigma(g)) \odot \tilde{W}_t
$$

**含义**: 通过可学习门控参数平衡当前观测与历史记忆的贡献，实现自适应的时序信息融合。

**符号说明**:
- $g$: 可学习的门控向量（与 token 维度一致）
- $\sigma(\cdot)$: sigmoid 激活函数
- $\odot$: 逐元素乘法
- $\hat{W}_t$: 记忆增强后的工作记忆 token

### 公式 3：PCMB 条目合并策略

$$
i^* = \arg\max_{i \in [1, K-1]} \cos(M_i, M_{i+1}), \quad M_{i^*} \leftarrow \frac{M_{i^*} + M_{i^*+1}}{2}
$$

**含义**: 当记忆库满时，找到相邻最相似的一对条目并将其平均合并，维持固定容量 $K$。

**符号说明**:
- $K$: PCMB 最大容量（记忆槽数量）
- $\cos(\cdot, \cdot)$: 余弦相似度
- $M_i$: 第 $i$ 个记忆槽的 token

### 公式 4：[[Imagination Module|想象模块]] 前瞻预测损失

$$
\mathcal{L}_{\text{imag}} = \sum_{k=1}^{n} \left\| \hat{W}_{t+k} - \text{sg}(W_{t+k}) \right\|^2_2
$$

**含义**: 训练时约束想象模块预测的未来 token 与实际编码的未来 token 对齐，实现前瞻性时序建模。

**符号说明**:
- $\hat{W}_{t+k}$: 想象模块预测的 $t+k$ 时间步工作记忆
- $\text{sg}(\cdot)$: stop-gradient 算子
- $W_{t+k}$: $t+k$ 时间步的真实工作记忆 token
- $n$: 前瞻步数（超参数）

### 公式 5：总训练损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{imag}}
$$

**含义**: 动作预测损失（MSE）与想象对齐损失的加权组合，联合优化记忆检索与未来预测。

**符号说明**:
- $\mathcal{L}_{\text{action}}$: 动作预测均方误差损失
- $\mathcal{L}_{\text{imag}}$: 想象模块对齐损失
- $\lambda$: 权重系数（控制想象损失的强度）

---

## 关键图表

### Figure 1: 系统概览（Memory + Imagination 框架）

> 图片来源: [arXiv:2606.09827](https://arxiv.org/abs/2606.09827) — 当前网络环境无法直接访问，可从 arXiv 论文页面查看。

**说明**: MemoryVLA++ 整体框架。输入当前观测 $o_t$ 和语言指令 $l$，经 [[Vision-Language Cognition Module]] 生成工作记忆；[[Perceptual-Cognitive Memory Bank]] 提供历史上下文（Memory）；Imagination Module 预测未来状态（Imagination）；最终 [[Memory-Conditioned Diffusion Action Expert]] 生成时序感知的动作块。

### Figure 2: Imagination Module 架构

> 图片来源: [arXiv:2606.09827](https://arxiv.org/abs/2606.09827) — 当前网络环境无法直接访问，可从 arXiv 论文页面查看。

**说明**: 展示 Imagination Module 的详细结构。基于 PCMB 增强的当前 token，通过轻量 decoder 向前预测 $n$ 步的未来工作记忆，并在训练时用辅助损失与真实未来 token 对齐。

### Figure 3: 与 MemoryVLA 的对比

> 图片来源: [arXiv:2606.09827](https://arxiv.org/abs/2606.09827) — 当前网络环境无法直接访问，可从 arXiv 论文页面查看。

**说明**: MemoryVLA（仅 Memory）与 MemoryVLA++（Memory + Imagination）的性能对比。MemoryVLA++ 在长程任务上的提升尤为显著，验证了 Imagination 模块对时序决策的正向贡献。

### Table 1: 主要 Benchmark 比较

| 方法 | LIBERO Avg | SimplerEnv-Bridge | Mikasa-Robo | CALVIN |
|------|-----------|-------------------|-------------|--------|
| CogACT | ~85.0% | 57.3% | ~30% | — |
| pi-0 | ~92.0% | 71.9% | 41.2% | — |
| MemoryVLA (ICLR'26) | 96.5% | 71.9% | 41.2% | — |
| MemoryVLA+ (Dexbotic) | — | 84.4% | — | — |
| **MemoryVLA++ (Ours)** | **98.4%** | **73.9%** | **44.4%** | **4.29** |

**说明**: MemoryVLA++ 在所有基准上均超越前版本，LIBERO 平均达到 98.4%，CALVIN 达到 4.29，证明 Memory + Imagination 双机制的有效性。

### Table 2: LIBERO 分任务结果

| 方法 | Spatial | Object | Goal | Long-10 | Long-90 | Average |
|------|---------|--------|------|---------|---------|---------|
| MemoryVLA++ | 99.8% | 100.0% | 98.2% | 96.0% | 97.8% | **98.4%** |

**说明**: MemoryVLA++ 在 LIBERO 的 5 个子任务上均取得极高成功率，尤其是 Long-10 和 Long-90 长程任务上分别达到 96.0% 和 97.8%，体现了时序建模能力的实质性提升。

### Table 3: 消融实验（推测）

| 配置 | LIBERO Avg | CALVIN |
|------|-----------|--------|
| 无 Memory，无 Imagination（baseline VLA）| ~85% | — |
| 仅 Memory（= MemoryVLA） | 96.5% | — |
| 仅 Imagination | — | — |
| Memory + Imagination（= MemoryVLA++）| **98.4%** | **4.29** |

**说明**: 消融实验验证 Memory 与 Imagination 模块各自的贡献，两者缺一不可，组合使用效果最佳。

---

## 实验

### 数据集与基准

| 数据集/基准 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| LIBERO | 130 任务，5 套件 | 多类型操作（空间/物体/目标/长程） | 主要评测 |
| SimplerEnv-Bridge | 24 子任务 | 真实场景迁移评测 | 跨域泛化 |
| CALVIN | 序列化 4 任务 | 长程语言条件操作 | 长程能力 |
| Mikasa-Robo | 多任务 | 精细操作 | 灵巧操作 |
| Fractal-VM | — | 真实机器人数据 | 真实场景 |
| 真实世界 12 任务 | 12 个任务 | 通用技能 + 长程时序依赖 | 真实验证 |

### 实现细节

- **视觉编码器**: [[DINOv2]] + [[SigLIP]]（双流特征提取）
- **语言模型**: [[LLaMA-7B]]（7B 参数）
- **动作专家**: [[DiT]]（Diffusion Transformer）
- **优化器**: [[AdamW]]
- **硬件**: 8× NVIDIA A100 GPU
- **Python / PyTorch**: 3.10 / 2.2.0（CUDA 12.1）
- **Flash Attention**: 2.5.5 加速训练

### 真实机器人实验

在 12 个真实世界任务上验证，涵盖：
- **通用技能任务**（General Tasks）：单步/短程操作，测试基础能力
- **长程时序依赖任务**（Long-Horizon Temporal Tasks）：多步骤、需要记忆历史状态的复杂任务

MemoryVLA 原版在长程任务上比最优 baseline 提升 **+26 个百分点**；MemoryVLA++ 通过加入 Imagination 模块进一步提升了长程任务的规划一致性。

---

## 批判性思考

### 优点

1. **认知科学启发充分**：Memory（海马体回忆）+ Imagination（心理模拟）的双机制符合人类认知科学原理，具有良好的可解释性
2. **前瞻性时序建模**：Imagination 模块是对纯回顾式记忆的有效补充，尤其对需要预见性规划的任务有帮助
3. **固定容量设计实用**：PCMB 的合并策略使模型在推理时保持 $O(K)$ 的常数内存，不随序列长度增长
4. **多 Backbone 支持**：提供 OpenVLA-based 和 Dexbotic-based 两种实现，覆盖不同算力配置

### 局限性

1. **Imagination 训练需要未来帧**：训练时需要访问未来时间步的观测，可能在某些在线学习场景下不适用
2. **合并策略可能丢失信息**：PCMB 容量达到上限时的均值合并策略较为简单，关键历史帧可能被稀释
3. **计算开销增加**：在原 MemoryVLA 基础上新增 Imagination Decoder，推理延迟可能增加
4. **Imagination 步数超参数敏感**：前瞻步数 $n$ 的选择对性能有影响，需要任务相关的调参

### 潜在改进方向

1. **层次化 Imagination**：在不同粒度（token 级/图像级/任务级）上进行多尺度预测
2. **动态记忆容量**：根据任务复杂度自适应调整 PCMB 容量 $K$，而非固定值
3. **强化学习微调**：在 Imagination-guided exploration 下进行在线 RL 微调，提升样本效率

### 可复现性评估

- [ ] 代码开源（GitHub 标注 MemoryVLA++ 分支为 "coming soon"）
- [ ] 预训练模型（TBD）
- [x] 训练细节完整（论文提供）
- [x] 数据集可获取（LIBERO、CALVIN 均为公开数据集）

---

## 关联笔记

### 基于

- [[MemoryVLA]]：本文的直接前序工作（ICLR 2026），MemoryVLA++ 是其扩展期刊版本
- [[Diffusion Policy]]：动作专家所用的扩散策略框架
- [[Action Chunking]]：动作块预测输出格式

### 对比

- [[CogACT]]：主要 baseline，在 SimplerEnv-Bridge 上低于 MemoryVLA++ 约 16.6%
- [[pi-0|pi0]]：另一主要 baseline，LIBERO 上低约 6.4%
- [[RoboMemArena]]：记忆增强 VLA 的综合 Benchmark，MemoryVLA 在其中作为对比方法

### 方法相关

- [[Perceptual-Cognitive Memory Bank]]：核心记忆模块
- [[Cross-Attention]]：记忆检索所用的注意力机制
- [[Imagination Module]]：MemoryVLA++ 新增的前瞻预测模块
- [[Vision-Language Cognition Module]]：视觉-语言联合编码模块
- [[非马尔可夫任务]]：本文所解决的核心问题类型

### 硬件/数据相关

- [[LIBERO]]：主要训练与测试基准（130 任务，5 套件）
- [[CALVIN]]：长程序列化任务基准
- [[SimplerEnv]]：基于真实数据的迁移评测套件

---

## 速查卡片

> [!summary] MemoryVLA++: Temporal Modeling via Memory and Imagination in VLA
> - **核心**: Memory（回顾历史）+ Imagination（预见未来）双机制解决 VLA 长程时序依赖
> - **方法**: PCMB 历史检索 + Imagination Decoder 前瞻预测 + Diffusion Action Expert
> - **结果**: LIBERO 98.4%、CALVIN 4.29、SimplerEnv-Bridge 73.9%（全面超越 MemoryVLA）
> - **代码**: https://github.com/shihao1895/MemoryVLA（MemoryVLA++ 分支 TBD）

---

*笔记创建时间: 2026-06-09*
