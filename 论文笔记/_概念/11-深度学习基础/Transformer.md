---
type: concept
aliases: [Transformer, 变换器, 注意力模型]
---

# Transformer

## 定义

Transformer 是 Vaswani et al. (2017) 提出的基于自注意力机制的序列建模架构，已成为 NLP、CV、机器人学等领域的核心基础架构。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## 核心要点

1. 完全基于注意力机制，摒弃 RNN 的序列依赖，支持并行计算
2. 多头注意力（Multi-Head Attention）从多个子空间捕捉不同类型的依赖关系
3. 位置编码（Positional Encoding / RoPE）弥补注意力的位置无关性
4. 编码器-解码器结构或仅编码器/仅解码器的变体广泛应用

## 代表工作

- [[EA-WM]]: 基于 Transformer 的扩散架构用于机器人视频生成

## 相关概念

- [[交叉注意力]]
- [[LoRA]]
- [[DiT]]
