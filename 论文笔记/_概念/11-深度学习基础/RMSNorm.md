---
type: concept
aliases: [Root Mean Square Norm, Root Mean Square Layer Normalization]
---

# RMSNorm

## 定义

一种简化版的归一化层，只用张量的 RMS（均方根）做缩放，不像 [[Batch Normalization|LayerNorm]] 那样减均值，因此参数和计算量都更小、对训练稳定性影响相近。

## 数学形式

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

## 核心要点

1. 不做去均值，因此对协变量偏移敏感性略高，但训练效果通常与 LayerNorm 相当
2. 计算更轻（无需均值统计），适合大模型大上下文
3. 是 LLaMA、PaLM、Mistral 等主流大模型的默认归一化
4. 在 [[线性注意力]] / [[Gated DeltaNet]] 中常用于 Key 归一化以约束谱范数

## 代表工作

- [[SANA-WM]]: 在 GDN 的 Key 路径上做 RMSNorm + ReLU + $1/\sqrt{D \cdot S}$ 缩放

## 相关概念

- [[Batch Normalization]]
- [[Transformer]]
- [[Gated DeltaNet]]
