---
type: concept
aliases: [QT-block, Query Transformer Block, Query-based Transformer]
---

# Query Transformer

## 定义

以一组可学习 query token 通过交叉注意力从条件特征序列中聚合信息的 Transformer 模块；query 维数通常远小于条件序列长度，实现高效的条件特征压缩与提取。

## 数学形式

$$
Q = W_Q \cdot q_{\text{learn}}, \quad K = W_K \cdot x_{\text{cond}}, \quad V = W_V \cdot x_{\text{cond}}
$$

$$
\text{output} = \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

其中 $q_{\text{learn}}$ 为可学习 query，$x_{\text{cond}}$ 为条件特征（如视觉/语言 token）。

## 核心要点

1. **固定输出维度**: 无论条件序列多长，输出 token 数等于 query 数，适合动作生成等需要固定输出长度的任务
2. **计算高效**: 相比全序列自注意力，Query Transformer 的复杂度与条件序列长度呈线性而非平方关系
3. **广泛应用于机器人策略**: Diffusion Policy、π0、RoboFlamingo 等均采用此结构聚合视觉-语言特征

## 代表工作

- [[omega-EVA]]: Stage 2 Flow 策略使用 12 个 QT 块提取 dynamics-aware 当前特征并生成动作提议
- [[Diffusion Policy]]: 使用 Query Transformer 聚合多视角视觉特征
- Flamingo: 最早将 QT 结构（Perceiver Resampler）用于多模态特征压缩

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[11-深度学习基础]]
