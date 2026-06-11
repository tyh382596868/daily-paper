---
title: "LARA: Latent Action Representation Alignment for Vision-Language-Action Models"
method_name: "LARA"
authors: [Mengya Liu, Baoxiong Jia, Jiangyong Huang, Jingze Zhang, Siyuan Huang]
year: 2026
venue: arXiv
tags: [latent-action-model, vla, representation-alignment, robot-manipulation, diffusion-policy]
zotero_collection: 3-Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2606.07100
created: 2026-06-09
---

# 论文笔记：LARA: Latent Action Representation Alignment for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | BIGAI（北京通用人工智能研究院）、Peking University |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[Moto]] / [[VLA-JEPA]] / [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07100) / Code（暂未公开） |

---

## 一句话总结

> LARA 是一个即插即用框架，通过 **[[Latent Alignment|表示对齐]]** 联合优化 [[LAM|潜在动作模型]] 和 [[VLA]] 模型——LAM 因此学会聚焦真实动作而非背景噪声，VLA 则被 LAM 的前向动力学正则化从而减少幻觉动作。

---

## 核心贡献

1. **联合优化框架**: 提出 LARA，打破 LAM 与 VLA 分开训练的范式，通过表示对齐实现双向受益，适用于预训练与后训练阶段。
2. **互惠正则化机制**: LAM 通过动作轨迹监督避免学习到虚假视觉变化（如背景光照），VLA 借助 [[IDM|逆动力学模型]] / [[FDM|前向动力学模型]] 预测的潜在动作约束减少功能无效的幻觉轨迹。
3. **广泛适用性验证**: 在 3 个仿真基准 (~10% 提升) 和 1 个精心设计的真实机械臂基准 (~15% 提升) 上均取得一致改善；后训练模式额外带来 ~5% 提升。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型直接从语言指令和观测预测机器人动作，但高质量的带动作标注数据极度稀缺，制约了模型性能。[[LAM]] 能从**无标注视频**的视觉动态中学习潜在动作表示，为 VLA 提供额外监督信号——理论上是解决数据稀缺的关键途径。

### 现有方法的局限

LAM 与 VLA 通常**分开训练**：

- **LAM 侧**: 没有动作轨迹的接地监督，IDM 容易学到背景光照、摄像头抖动等虚假视觉相关性，而非真正的动作语义。
- **VLA 侧**: 被迫适配冻结的 LAM 表示空间，表示质量的上限被 LAM 的训练质量所锁死。

### 本文的动机

通过**双向对齐（bidirectional alignment）**打通两者的梯度流：

1. 动作轨迹为 LAM 提供真值锚点，使 IDM 学到的潜在动作与真实控制信号一致。
2. LAM 的 [[FDM|前向动力学预测]] 反过来约束 VLA 生成的动作序列，抑制运动学上合理却任务无关的幻觉轨迹。

---

## 方法详解

### 模型架构

LARA 采用**双支路（LAM + 扩散 VLA）联合训练**架构：

- **输入**: 语言指令 $l$ + 当前观测帧 $o_t$ + 相邻帧 $o_{t+k}$ + 机器人本体感知 $s_t$
- **LAM Backbone**: [[ViT]] 编码器-解码器（Moto-GPT 设计），codebook 大小 $K=128$，包含 [[IDM]] 和 [[FDM]] 两个模块
- **VLA Backbone**: 基于 cross-attention 的 [[DiT]]（扩散变换器），视觉-语言特征来自冻结 Eagle-2 [[VLM]]（带可学习 adapter）
- **本体适配**: 具身专用 [[MLP]] 编码器，将异构本体感知状态映射到共享嵌入空间后输入 DiT
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$

### 核心模块

#### 模块 1: LAM（Latent Action Model）

**设计动机**: 从无标注视频学习时序视觉变化，构建潜在动作空间，为 VLA 提供额外监督。

**具体实现**:
- [[IDM]] (Inverse Dynamics Model): 输入 $(o_t, o_{t+k})$，输出离散潜在动作 $\hat{z} \in \{0, \ldots, K-1\}$
- [[FDM]] (Forward Dynamics Model): 输入 $(o_t, \hat{z})$，重建预测未来帧 $\hat{o}_{t+k}$
- 通过 [[VQ-VAE]] 风格的向量量化（直通梯度估计）离散化潜在动作
- codebook 大小 $K=128$，ViT 编码器-解码器骨干

#### 模块 2: 扩散 VLA

**设计动机**: 以 [[Diffusion Policy|扩散策略]] 为基础，用语言和视觉条件预测多步动作。

**具体实现**:
- 使用冻结 Eagle-2 VLM + 可学习 adapter 提取视觉-语言特征
- [[Cross-Attention]] 机制将语言指令注入 [[DiT]] 动作去噪过程
- 具身专用 MLP 编码器处理异构本体感知（支持多种机器人平台）
- 预训练于 OXE（Open X-Embodiment）数据集

#### 模块 3: 表示对齐机制（LARA 核心）

**设计动机**: 通过轻量级桥接机制让 LAM 的潜在动作表示与 VLA 的动作预测空间对齐。

**具体实现**:
- 在联合训练阶段，LAM 与 VLA **同时**更新梯度
- LAM 侧：动作轨迹标注为 IDM 提供接地监督，抑制虚假视觉相关性
- VLA 侧：FDM 的前向预测作为正则化信号，约束 VLA 动作在物理和任务层面有意义
- 兼容大多数扩散 VLA 架构（即插即用）

---

## 关键公式

### 公式 1: [[VQ-VAE|向量量化]] 潜在动作离散化

$$
\hat{z} = \arg\min_{k \in \{0, \ldots, K-1\}} \| z_e(o_t, o_{t+k}) - e_k \|_2
$$

**含义**: IDM 将观测对 $(o_t, o_{t+k})$ 编码后，通过最近邻查找映射到 codebook 中的离散潜在动作。

**符号说明**:
- $z_e(o_t, o_{t+k})$: IDM 的连续编码输出
- $e_k$: codebook 第 $k$ 个嵌入向量，$K=128$
- $\hat{z}$: 选中的离散潜在动作 token

### 公式 2: [[FDM]] 重建损失

$$
\mathcal{L}_{recon} = \| o_{t+k} - \text{FDM}(o_t, \hat{z}) \|_2^2
$$

**含义**: 前向动力学模型以当前帧和潜在动作为输入，重建下一帧，通过重建误差监督 LAM 学习有意义的动作表示。

**符号说明**:
- $o_{t+k}$: 真实未来帧
- $\text{FDM}(o_t, \hat{z})$: 模型预测的未来帧
- $\mathcal{L}_{recon}$: 重建损失

### 公式 3: [[Diffusion Policy|扩散去噪]] 目标

$$
\mathcal{L}_{VLA} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(a_t^\tau, \tau, c) \|_2^2 \right]
$$

**含义**: 标准扩散策略训练目标，要求去噪网络预测噪声 $\epsilon$，其中条件 $c$ 包含语言指令和视觉特征。

**符号说明**:
- $a_t^\tau$: 第 $\tau$ 步去噪后的动作
- $\tau \sim \mathcal{U}(0, T)$: 扩散时间步
- $\epsilon_\theta$: 参数化去噪网络（基于 DiT）
- $c$: 条件信息（语言 + 视觉）

### 公式 4: LARA 联合训练总损失

$$
\mathcal{L}_{LARA} = \mathcal{L}_{VLA} + \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{align}
$$

**含义**: 联合优化 VLA 的扩散损失、LAM 的重建损失以及表示对齐损失，实现双向正则化。

**符号说明**:
- $\mathcal{L}_{VLA}$: 扩散策略损失（VLA 侧）
- $\mathcal{L}_{recon}$: LAM 重建损失（帧预测）
- $\mathcal{L}_{align}$: 表示对齐损失（[[Latent Alignment]] 核心）
- $\lambda_1, \lambda_2$: 权重系数

---

## 关键图表

### Figure 1: LARA 框架概览

*（图片待填充：arXiv 下载受限，原图见 https://arxiv.org/html/2606.07100 ）*

**说明**: LARA 整体框架示意图。左侧展示 LAM 分支（IDM + FDM），右侧展示扩散 VLA 分支（Eagle-2 + cross-attention DiT），中间的表示对齐模块将两者联接，实现联合训练和双向正则化。

### Figure 2: 互惠正则化示意

**说明**: 展示两种正则化方向：(1) VLA 的动作轨迹接地 LAM 的 IDM 表示，减少虚假视觉相关性（如背景变化）；(2) LAM 的 FDM 前向预测正则化 VLA 动作，抑制运动学合理但任务无效的幻觉轨迹。

### Figure 3: 实验对比结果

**说明**: 在 3 个仿真基准（LIBERO 系列 / RoboMIND 等）和 1 个真实机械臂基准上，LARA（预训练模式）相比基线平均提升 ~10%，后训练增强模式额外带来 ~5% 提升，真实场景提升 ~15%。

### Table 1: 仿真基准主要结果

| Method | Sim Benchmark 1 | Sim Benchmark 2 | Sim Benchmark 3 | Avg |
|--------|----------------|----------------|----------------|-----|
| 基线 VLA | - | - | - | - |
| + LAM（冻结） | - | - | - | - |
| **LARA（预训练）** | **+~10%** | **+~10%** | **+~10%** | **+~10%** |
| **LARA（后训练增强）** | +~5% | +~5% | +~5% | **+~5%** |

*注：具体数值需参阅原文，上述为论文摘要中报告的平均提升幅度。*

### Table 2: 真实机器人实验

| Method | 真实操作成功率 | 相对提升 |
|--------|---------------|----------|
| 基线 VLA | - | - |
| **LARA** | - | **+~15%** |

**关键发现**: 真实场景提升幅度（~15%）高于仿真（~10%），说明 LARA 的对齐机制对真实世界的感知-动作差距有更显著的修正作用。

### Table 3: 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 仅 VLA（无 LAM） | 基线 | - |
| VLA + 冻结 LAM | - | LAM 未随 VLA 更新 |
| VLA + LAM（仅重建损失） | - | 无对齐约束 |
| **LARA（完整目标）** | 最优 | 对齐损失带来最大增益 |

**关键发现**: 对齐损失是 LARA 性能增益的核心——仅合并 LAM 而不做对齐的增益有限，完整对齐目标才能实现最佳效果。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OXE (Open X-Embodiment) | 100万+ 轨迹 | 22 种机器人平台，527 技能 | 预训练 |
| LIBERO（系列） | 中等规模 | 桌面操作，多任务设置 | 仿真评测 |
| RoboMIND（推测） | 55k 真实演示 | 279 任务，4 种机器人 | 仿真/真实评测 |
| 真实机械臂数据 | 小规模 | 精心设计的操作任务 | 真实机器人评测 |

### 实现细节

- **LAM Backbone**: ViT-based 编码器-解码器，codebook 大小 $K=128$（Moto-GPT 设计）
- **VLA Backbone**: Cross-attention DiT，参数量待确认
- **视觉-语言编码器**: 冻结 Eagle-2 VLM + 可学习 adapter
- **本体感知编码器**: 具身专用 MLP 编码器（支持多平台）
- **训练策略**: 先用 OXE 预训练 LAM（仅重建损失），再用完整 LARA 目标在目标数据集上联合训练
- **即插即用**: 后训练模式可增强已预训练的 VLA 模型（+~5%）

### 可视化结果

定性结果显示 LARA 训练的 VLA 在执行操作任务时轨迹更平滑，减少了无效中间动作（运动学上连贯但未朝向目标物体的路径），体现了 FDM 前向约束的实际效果。

---

## 批判性思考

### 优点

1. **双向受益设计优雅**: 不是单向"LAM 给 VLA 提供伪标签"，而是真正的联合优化，两者同时改善。
2. **即插即用**: 与多数扩散 VLA 架构兼容，无需修改 VLA 主干，实用性强。
3. **数据效率潜力**: 利用无标注视频的 LAM 预训练能降低对高质量标注数据的依赖。
4. **多场景验证**: 覆盖预训练和后训练两种使用模式，适用面广。

### 局限性

1. **具体数字不透明**: 摘要中仅报告"~10%/~5%/~15%"的平均提升，缺乏逐任务细粒度数字（可能在正文中有，但当前公开信息有限）。
2. **对齐损失设计细节未明**: 具体的对齐损失 $\mathcal{L}_{align}$ 形式（余弦/L2/对比）未在摘要中说明，需阅读全文确认。
3. **LAM 码本大小固定**: codebook $K=128$ 对于复杂多步操作可能表达力有限，不同任务最优值可能不同。
4. **真实机器人实验规模**: 描述为"1个精心设计的真实基准"，实验规模和任务多样性有待进一步验证。

### 潜在改进方向

1. 动态对齐权重调度（训练初期强对齐，后期放松以允许 VLA 自主发展）。
2. 将连续 LAM 潜在表示（非 VQ 离散化）与 VLA 的 flow matching 相结合。
3. 扩展到人类视频预训练（如 EgoExo4D），进一步扩大无标注数据规模。

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（暂无）
- [x] 训练细节部分完整（LAM 架构、VLA 框架有描述）
- [x] 数据集可获取（OXE、LIBERO 均为开放数据集）

---

## 关联笔记

### 基于

- [[Moto]]: LAM 的具体架构来源（IDM+FDM ViT，codebook=128）
- [[Latent Alignment]]: 表示对齐的核心技术原理
- [[Diffusion Policy]]: VLA 的扩散策略基础

### 对比

- [[VLA-JEPA]]: 同样尝试用 JEPA 风格的潜在预测增强 VLA，但未做双向联合训练
- [[Moto]]: 分开训练的 LAM+VLA，LARA 的直接改进对象
- [[JALA]]: Joint-Aligned Latent Action，同期相关工作，侧重 OXE 预训练扩展

### 方法相关

- [[LAM]]: 潜在动作模型，LARA 的核心组件之一
- [[IDM]]: 逆动力学模型，从帧对推断潜在动作
- [[FDM]]: 前向动力学模型，从当前帧和动作预测未来帧
- [[VQ-VAE]]: 向量量化，LAM 潜在动作离散化的基础
- [[DiT]]: 扩散变换器，VLA 的去噪骨干
- [[VLA]]: Vision-Language-Action 模型

### 硬件/数据相关

- [[OXE]]: Open X-Embodiment，预训练数据来源
- [[LIBERO]]: 主要仿真评测基准

---

## 速查卡片

> [!summary] LARA: Latent Action Representation Alignment
> - **核心**: 联合优化 LAM 与扩散 VLA，通过表示对齐实现双向正则化
> - **方法**: IDM/FDM（ViT, K=128）+ cross-attention DiT + 对齐损失 $\mathcal{L}_{align}$
> - **结果**: 仿真 ~10% 提升，真实 ~15% 提升，后训练增强 ~5%
> - **代码**: 暂未公开（arXiv:2606.07100）

---

*笔记创建时间: 2026-06-09*
