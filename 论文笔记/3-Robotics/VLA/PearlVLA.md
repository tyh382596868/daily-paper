---
title: "PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Space"
method_name: "PearlVLA"
authors: [Bochen Yang, Lianlei Shan]
year: 2026
venue: arXiv
tags: [vla, latent-space-deliberation, world-model, action-refinement, process-reward-rl]
zotero_collection: 3-Robotics/VLA
image_source: pending
arxiv_html: https://arxiv.org/html/2606.17924v1
created: 2026-06-21
---

# 论文笔记：PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Space

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Imperial College London（Bochen Yang）；Tsinghua University（Lianlei Shan）|
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[OpenVLA-OFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17924) / Code 暂未公开 |

---

## 一句话总结

> PearlVLA 将 VLA 的"审议"过程搬入 VLM 潜空间，用轻量冻结潜空间世界模型的未来反馈迭代精炼隐式动作计划，同时保留低延迟并行动作块输出，在 LIBERO 上达到 98.7% 的 SOTA。

---

## 核心贡献

1. **潜空间渐进精炼（PEARL）**: 将 VLM meta-query 分为固定视觉接地分支和迭代潜在计划分支，在 K 轮精炼中用计划条件世界查询驱动 [[RefineNet]] 做 scheduled residual 更新，无需解码为文字/像素。
2. **轻量冻结潜空间世界模型**: 每轮精炼用"计划条件世界查询"探测一个轻量级冻结 [[Latent World Model]]，获取动作无关的未来观测潜变量反馈，避免像素级重建开销。
3. **因果精炼分组过程奖励 RL（CRG-PRL）**: 在监督训练之上引入 [[Causal Refinement-Grouped Process-Reward RL]]，用组相对优势对同状态下的计划编辑排序，以更长视野的想象未来奖励优化精炼轨迹。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在"低延迟控制"与"显式审议"之间存在两难：

- **直接解码**（如 [[OpenVLA-OFT]]）：从视觉语言主干直接映射到动作块，延迟低但缺乏多步规划能力。
- **显式推理**（CoT 文字链、像素级子目标、候选动作搜索）：可提升规划质量，但带来大量延迟与算力开销。

### 现有方法的局限

| 方法类型 | 代表 | 局限 |
|----------|------|------|
| 直接动作解码 | [[OpenVLA-OFT]] | 无审议，复杂任务规划弱 |
| 文字/视觉 CoT | [[CoT-VLA]] | 额外文字 token 推理开销大 |
| 像素级子目标 | [[DreamVLA]] | 需像素重建，计算昂贵 |
| 候选动作搜索 | Value VLA | 需多次前向推理 |
| flow-matching 动作头 | [[pi0]] | 迭代采样引入推理延迟 |

### 本文的动机

审议不一定要在像素空间或文字空间进行——VLM 的潜空间已经编码了丰富的视觉-语言语义，在此空间用轻量世界模型做多轮反馈精炼，既可获得审议带来的规划提升，又可保留并行动作块解码的低延迟优势。

---

## 方法详解

### 模型架构

PearlVLA 在 [[OpenVLA-OFT]] 基础上插入潜空间精炼模块，整体架构：

- **输入**: 语言指令 $l$ + 观测 $o_t$ + 机器人状态历史（proprioceptive tokens）
- **Backbone**: [[SigLIP]] + [[DINOv2]] 融合视觉特征 → [[LLaMA-2]] 7B 语言主干（[[LoRA]] 微调）
- **Proprioceptive Encoder**: Transformer-based，将状态历史编码为 proprioceptive tokens 插入多模态序列
- **Meta-Query 分离**: VLM 输出的 meta-query 分为：
  - **视觉接地分支**（固定）：提供稳定的场景感知
  - **潜在计划分支**（迭代）：进行 K 轮渐进精炼
- **潜空间精炼**：K=4 轮 [[RefineNet]] 更新（默认配置，27.5 Hz）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$（并行连续动作解码，regression head）
- **总参数**: 基于 OpenVLA-7B

### 核心模块

#### 模块1: 潜在计划分支 + RefineNet

**设计动机**: 在 [[VLM Latent Space Deliberation]] 中迭代精炼，避免直接解码到文字/像素的开销

**具体实现**:

- 初始化：VLM 生成粗粒度语义草稿（coarse semantic draft）$z^{(0)}$
- 每轮精炼 $k = 1, \ldots, K$：
  1. 当前潜在计划 $z^{(k-1)}$ 条件化生成**计划条件世界查询**（plan-conditioned world query）$q^{(k)}$，维持在锚定（anchored）的冻结 WM 条件空间内
  2. 冻结 [[Latent World Model]] 用 $q^{(k)}$ 推断**动作无关未来观测潜变量** $\hat{o}_{t+1}^{(k)}$（action-free future observation latent）
  3. [[RefineNet]] 将 $\hat{o}_{t+1}^{(k)}$ 转化为对潜在计划的**残差修正** $\Delta z^{(k)}$，执行 scheduled residual write-back：$z^{(k)} = z^{(k-1)} + \alpha_k \cdot \Delta z^{(k)}$（$\alpha_k$ 为 scheduling 系数）
  4. 下一轮的世界查询随更新后的计划变化，而非复用静态未来特征
- 辅助对齐损失：将精炼过程对齐到冻结 WM 的条件空间，确保训练稳定

#### 模块2: 冻结潜空间世界模型（Frozen Latent World Model）

**设计动机**: 提供动作无关的未来感知信号，无需像素重建

**具体实现**:
- 轻量级网络，训练后参数冻结
- 接收计划条件世界查询 $q^{(k)}$，输出未来帧的潜变量表示
- 类似 [[JEPA]] 风格：在特征空间预测而非像素空间预测
- 锚定世界查询（anchored world queries）机制：将每轮查询约束在世界模型的预训练条件空间内，防止分布外偏移

#### 模块3: CRG-PRL（因果精炼分组过程奖励 RL）

**设计动机**: 监督训练后，进一步用 RL 优化精炼轨迹本身

**具体实现**:
- **Causal Refinement-Grouped**: 对同一状态下的多个潜在计划编辑在组内比较（group-relative advantage），隔离每次编辑的因果效应，无需学习价值函数（value function）
- **Process Reward**: 每轮精炼结束后，将潜在滚动（latent rollout）解码为想象未来帧并评分，提供 per-round 过程奖励信号
- **longer-horizon 想象**: 用自回归扩展的想象未来帧奖励，激励精炼过程关注更长视野的规划质量

---

## 关键公式

### 公式1: [[Latent Plan Refinement|潜在计划残差更新]]

$$
z^{(k)} = z^{(k-1)} + \alpha_k \cdot \Delta z^{(k)}
$$

**含义**: 第 $k$ 轮精炼中，用 scheduled 系数 $\alpha_k$ 控制 RefineNet 输出的残差 $\Delta z^{(k)}$ 对当前潜在计划的更新幅度

**符号说明**:
- $z^{(k)}$: 第 $k$ 轮精炼后的潜在动作计划
- $z^{(0)}$: VLM 初始生成的粗粒度语义草稿
- $\alpha_k$: 第 $k$ 轮的 scheduling 系数（决定写入步长）
- $\Delta z^{(k)}$: RefineNet 基于未来反馈生成的残差修正

### 公式2: [[Latent World Model|计划条件未来推断]]

$$
\hat{o}_{t+1}^{(k)} = f_{\mathrm{WM}}\!\left(q^{(k)}\right), \quad q^{(k)} = g\!\left(z^{(k-1)}\right)
$$

**含义**: 从当前潜在计划 $z^{(k-1)}$ 生成计划条件世界查询，冻结世界模型据此推断动作无关的未来观测潜变量

**符号说明**:
- $f_{\mathrm{WM}}$: 冻结潜空间世界模型
- $q^{(k)}$: 第 $k$ 轮的计划条件世界查询
- $g(\cdot)$: 从潜在计划到世界查询的映射函数
- $\hat{o}_{t+1}^{(k)}$: 第 $k$ 轮的动作无关未来观测潜变量

### 公式3: [[Causal Refinement-Grouped Process-Reward RL|CRG-PRL 组相对优势]]

$$
A^{(k)}_i = r^{(k)}_i - \frac{1}{G}\sum_{j=1}^{G} r^{(k)}_j
$$

**含义**: 在同一精炼状态下对 G 个候选计划编辑计算组相对优势，仅比较同组内编辑的奖励，隔离因果效应

**符号说明**:
- $A^{(k)}_i$: 第 $k$ 轮第 $i$ 个编辑的优势估计
- $r^{(k)}_i$: 第 $i$ 个编辑对应的过程奖励（来自解码后的想象未来帧评分）
- $G$: 组内候选编辑数量
- 不依赖学习的价值函数，直接用组内均值作基线

### 公式4: [[VLM Latent Space Deliberation|K 轮精炼后的动作解码]]

$$
a_{t:t+H} = \pi_{\mathrm{head}}\!\left(z^{(K)}\right)
$$

**含义**: 经过 K 轮精炼后，用并行连续动作回归头将精炼后的潜在计划解码为动作块

**符号说明**:
- $z^{(K)}$: K 轮精炼后的最终潜在动作计划
- $\pi_{\mathrm{head}}$: 并行连续动作回归头（parallel regression head）
- $a_{t:t+H}$: 动作块（chunk size H，默认 H=10）
- $K$: 精炼轮数（默认 K=4）

---

## 关键图表

### Figure 1: VLA 设计选择对比

> 🖼️ **Figure 1: Design Choices for VLA Policies** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.17924v1)）

**说明**: 四种 VLA 设计范式对比：
- **(a) 直接动作解码**: VLM 表示 → 动作块，低延迟但无审议
- **(b) 文字/视觉推理 token**: 解码推理 token 后再解码动作，延迟大
- **(c) 世界模型评分候选动作**: 需多次前向推理，计算昂贵
- **(d) PearlVLA**: 在 VLM 潜空间做渐进精炼后并行解码动作块，兼顾审议与低延迟

### Figure 2: PearlVLA 完整架构图

> 🖼️ **Figure 2: PearlVLA Architecture** — 图片暂缺（arXiv 抓取失败，原图见 [arXiv HTML](https://arxiv.org/html/2606.17924v1)）

**说明**: 完整方法架构，展示 meta-query 分离、K 轮 RefineNet 精炼循环、冻结潜空间世界模型、proprioceptive encoder 以及最终并行动作解码头。

### Table 1: LIBERO Benchmark 主要结果

| 方法 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| OpenVLA | — | — | — | — | ~76% |
| OpenVLA-OFT | — | — | — | — | 97.1% |
| **PearlVLA (Supervised)** | — | — | — | — | **98.5%** |
| **PearlVLA + CRG-PRL** | — | — | — | — | **98.7%** |

**说明**: 对比 [[OpenVLA-OFT]] 主要基线，监督版 PearlVLA 将平均成功率从 97.1% 提升至 98.5%；CRG-PRL 阶段进一步提升至 98.7%，表明精炼轨迹在监督训练后仍有优化空间。

> 注：各 LIBERO 子集的精确数字未能从搜索结果中获取，完整数据见 [arXiv 原文](https://arxiv.org/abs/2606.17924)

### Table 2: 消融实验

| 配置 | LIBERO Avg | 说明 |
|------|-----------|------|
| K=0（无 RefineNet 精炼） | < 98.5% | 验证精炼轮次的作用 |
| K=4（默认） | 98.5% | 最佳精炼轮数，27.5 Hz |
| K=4 + CRG-PRL | 98.7% | RL 进一步优化精炼轨迹 |
| H=20（更长 chunk，K=4） | ~97% | 68.5 Hz，中等成功率下降 |
| 单一跨任务策略 | 98.1% | 对比套件特定策略的 98.5% |

**关键发现**:
- K=4 的精炼轮次在延迟（27.5 Hz 满足实时操作需求）和性能之间取得最佳平衡
- 增加 chunk size 到 H=20 可将吞吐提升至 68.5 Hz，但 LIBERO-Long 成功率下降 1.7 个点（PearlVLA 仅 1.7 点 vs 直接解码 3.2 点，说明潜空间精炼对 Long 任务更具鲁棒性）
- 单一跨套件策略（98.1%）与套件特定策略（98.5%）差距仅 0.4 个点，泛化能力强

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]]-Spatial | 10 tasks × 50 demos | 空间关系泛化 | 训练 + 测试 |
| [[LIBERO]]-Object | 10 tasks × 50 demos | 新物体类型泛化 | 训练 + 测试 |
| [[LIBERO]]-Goal | 10 tasks × 50 demos | 新目标条件泛化 | 训练 + 测试 |
| [[LIBERO]]-Long | 10 tasks × 50 demos | 长视野复合任务 | 训练 + 测试 |

### 实现细节

- **Backbone**: [[OpenVLA-7B]]（SigLIP + DINOv2 + LLaMA-2 7B）
- **微调方式**: [[LoRA]] 适配
- **精炼轮数**: K=4（默认），维持 27.5 Hz 实时吞吐
- **动作块大小**: H=10（默认），H=20（高吞吐配置）
- **动作头**: 并行连续动作回归头（非 flow-matching，避免迭代采样延迟）
- **训练阶段**: (1) 监督训练（SL）→ (2) CRG-PRL 强化学习微调
- **CRG-PRL**: 组相对优势，无需学习价值函数；用自回归扩展想象未来帧提供过程奖励

### 可视化结果

- PearlVLA 在长视野任务（LIBERO-Long）上相比直接解码基线的成功率下降幅度更小（1.7 vs 3.2 个点），说明潜空间审议对需要多步规划的任务尤其有益
- 单一跨任务策略仍保持 98.1% 平均成功率，LIBERO-Goal 成绩不变，LIBERO-Long 还略有提升

---

## 批判性思考

### 优点
1. **延迟友好的审议**: 所有精炼在潜空间完成，动作解码仍为并行一次性推理，实测 27.5 Hz 满足实时操作需求
2. **无需像素重建**: 世界模型工作在特征空间，避免了生成像素的计算开销（对比 DreamVLA 等方法）
3. **RL 与 SL 解耦**: CRG-PRL 在监督训练后独立优化精炼轨迹，无需重新训练 backbone

### 局限性
1. **图片信息有限**: 论文仅有 2 张图（21 页论文），图片描述依赖文字叙述
2. **per-suite 数字未公开**: 搜索可获得的数字以 Avg 为主，各子集细粒度结果需读原文
3. **World Model 预训练未详述**: 冻结潜空间 WM 的训练数据和具体架构在搜索结果中未获取

### 潜在改进方向
1. 将 CRG-PRL 推广到跨套件联合训练策略，进一步缩小单一策略与专用策略的差距
2. 探索自适应 K（根据任务难度动态调整精炼轮数），在简单任务上节省计算

### 可复现性评估
- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（论文中有较详细描述）
- [x] 数据集可获取（LIBERO 公开）

---

## 关联笔记

### 基于
- [[OpenVLA-OFT]]: PearlVLA 的直接基础，使用相同的 OpenVLA-7B 骨干和并行动作块解码
- [[OpenVLA]]: 上游 VLA 基础模型

### 对比
- [[OpenVLA-OFT]]: 主要基线，avg 97.1% → PearlVLA SL 98.5% → +CRG-PRL 98.7%
- [[CoT-VLA]]: 文字推理链方案，延迟更高
- [[DreamVLA]]: 像素级子目标方案，开销更大
- [[VLA-JEPA]]: 同为 latent world model + VLA 路线，但定位不同

### 方法相关
- [[VLM Latent Space Deliberation]]: PearlVLA 是该范式的代表工作
- [[Latent World Model]]: 核心组件，轻量冻结实现
- [[Action Chunking]]: 动作块输出机制
- [[GRPO]]: CRG-PRL 的精炼轨迹优化与 GRPO 的组相对优势思路相关
- [[JEPA]]: latent world model 的 JEPA 范式源头

### 硬件/数据相关
- [[LIBERO]]: 主要评测 benchmark（四个子集）

---

## 速查卡片

> [!summary] PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Space
> - **核心**: 在 VLM 潜空间做 K 轮渐进精炼，用冻结 latent world model 的未来反馈驱动 RefineNet 做 scheduled residual 更新
> - **方法**: Meta-query 分离 → K=4 轮 RefineNet 精炼 → 并行动作块解码 → CRG-PRL 强化优化精炼轨迹
> - **结果**: LIBERO 平均成功率 98.7%（SL: 98.5%，+CRG-PRL: 98.7%），实时吞吐 27.5 Hz
> - **代码**: 暂未公开，见 [arXiv](https://arxiv.org/abs/2606.17924)

---

*笔记创建时间: 2026-06-21*
