---
type: concept
aliases: [自回归建模, AR 建模, 自回归生成]
---

# Autoregressive Modeling

## 定义

一种生成建模范式，将联合分布 $P(x_1, \ldots, x_N)$ 通过链式法则分解为条件概率乘积，逐步生成序列中的每个元素。

## 数学形式

$$
P(x_1, \ldots, x_N) = \prod_{t=1}^{N} P(x_t \mid x_1, \ldots, x_{t-1})
$$

## 核心要点

1. **精确似然估计**: 可计算精确的对数似然，无需近似推断
2. **序列依赖捕捉**: 天然建模时序或位置依赖关系
3. **推理速度局限**: 逐 token 串行生成，推理速度受序列长度限制
4. **灵活条件生成**: 可直接做条件概率 $P(y \mid x)$（如给定语言生成动作）

## 代表工作

- [[GPT]]: 自然语言的自回归建模
- [[Autoregressive Transformer]]: 基于 Transformer 的自回归实现
- [[UniVLA]]: 多模态（视觉/语言/动作）统一自回归建模

## 相关概念

- [[Autoregressive Transformer]]: 主流实现架构
- [[Diffusion Policy]]: 另一类生成范式（非自回归）
- [[VQ-VAE]]: 配合自回归建模的离散化工具
