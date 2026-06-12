---
type: concept
aliases: [Multi-Head Attention, 多头注意力]
---

# MHA (Multi-Head Attention)

## 定义
Transformer 的核心组件，通过并行多个注意力头捕获不同子空间的依赖关系。

## 数学形式
$$\text{MHA}(Q,K,V) = \text{Concat}(h_1, ..., h_k)W^O$$
$$h_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## 核心要点
1. 多个独立注意力头并行计算
2. 不同头关注不同子空间的模式
3. 在 KinematicRL 等工作中用于多智能体注意力

## 相关概念
- [[Transformer]]
