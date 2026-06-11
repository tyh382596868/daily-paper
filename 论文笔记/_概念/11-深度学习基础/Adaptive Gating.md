---
type: concept
aliases: [自适应门控, 门控融合, adaptive gate, gate fusion]
---

# Adaptive Gating（自适应门控）

## 定义

一种用于动态平衡两路信息权重的融合机制：通过可学习的门控参数经 sigmoid 激活后逐元素调节两个输入的融合比例，而非固定权重相加或直接拼接。

## 数学形式

$$\hat{x} = \sigma(g) \odot x_1 + (1 - \sigma(g)) \odot x_2$$

其中 $g$ 为可学习门控向量，$\sigma(\cdot)$ 为 sigmoid，$\odot$ 为逐元素乘法。

## 核心要点

1. **软切换**：权重在 $[0, 1]$ 之间连续变化，比 hard attention 更稳定
2. **无需超参数**：融合比例完全由数据驱动，无需手动设定混合系数
3. **计算轻量**：仅增加一组与 token 维度相同的可学习参数
4. **在 VLA 记忆融合中的应用**：MemoryVLA 用它平衡"当前观测 token"与"历史记忆检索 token"的贡献

## 代表工作

- [[MemoryVLA]]（ICLR 2026）: 用于 PCMB 检索结果与当前工作记忆的融合
- [[MemoryVLA++]]（arXiv 2606.09827）: 继承并扩展

## 相关概念

- [[Cross-Attention]]
- [[Perceptual-Cognitive Memory Bank]]
