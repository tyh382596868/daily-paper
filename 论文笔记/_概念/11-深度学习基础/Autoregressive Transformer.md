---
type: concept
aliases: [自回归 Transformer, AR Transformer, Causal Transformer]
---

# Autoregressive Transformer

## 定义

一种基于 Transformer 的序列模型，通过因果（单向）注意力机制，以条件概率 $P(x_t \mid x_{<t})$ 的乘积形式自回归地生成 token 序列。

## 数学形式

$$
P(x_1, \ldots, x_N) = \prod_{t=1}^{N} P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

训练目标（最大似然）：

$$
\mathcal{L} = -\sum_{t=1}^{N} \log P_\theta(x_t \mid x_{<t})
$$

## 核心要点

1. **因果掩码（Causal Mask）**: 注意力矩阵对未来位置屏蔽，保证自回归性
2. **统一序列建模**: 任意模态（文本/图像/动作）均可表示为 token 序列联合建模
3. **Next-Token Prediction**: 训练目标简单统一，支持大规模无标注数据预训练
4. **生成能力**: 支持条件生成（语言→动作、语言→图像等）

## 代表工作

- [[GPT]]: 原始自回归语言模型
- [[Emu3]]: 视觉-语言统一 token 自回归模型
- [[UniVLA]]: 视觉-语言-动作统一离散 token 自回归 VLA 模型

## 相关概念

- [[VQ-VAE]]: 将连续信号离散化为 token 的基础工具
- [[Action Chunking]]: 动作预测的 token 粒度设计
- [[世界模型]]: 自回归 Transformer 的重要应用场景
