---
type: concept
aliases: [Autoregressive Transformer, AR Transformer, 自回归变换器, GPT-style Transformer]
---

# 自回归 Transformer

## 定义

基于 Transformer 架构的自回归生成模型，通过因果自注意力（causal self-attention）实现逐 token 的下一 token 预测，广泛用于语言、图像、视频等序列生成任务。

## 数学形式

$$
p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

## 核心要点

1. **因果掩码**: 通过上三角掩码矩阵确保每个位置只能关注其之前的 token，保证自回归性
2. **KV Cache**: 推理时缓存历史 K、V 矩阵，避免重复计算，线性时间推理
3. **Spatial-Temporal 扩展**: 在视频生成中，token 既有空间（空间 patch）维度又有时间（帧）维度，需要 Spatial-Temporal 注意力设计
4. **误差累积**: 自回归生成的已知挑战——每步错误会在后续步骤中放大（compounding errors）

## 代表工作

- [[RoboScape]]: 基于自回归 Transformer 进行动作条件化机器人视频生成，多任务联合训练
- [[Genie]]: 基于自回归 Transformer 的交互式世界模型

## 相关概念

- [[Causal Self-Attention]]: 自回归 Transformer 的核心组件
- [[VQ-VAE]]: 常与自回归 Transformer 配合，将连续图像/视频离散化为 token 序列
- [[自回归生成]]: 自回归 Transformer 实现的生成范式
