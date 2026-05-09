---
type: concept
aliases: [Exponential Moving Average, 指数移动平均]
---

# EMA

## 定义
用指数衰减的历史权重平均来平滑参数更新，常用于自监督学习中的 teacher 网络维护或扩散模型的权重平均。

## 数学形式
$$\theta'_t = \mu \cdot \theta'_{t-1} + (1-\mu) \cdot \theta_t$$
其中 $\mu \in (0,1)$ 为动量参数（通常 0.999），$\theta$ 为 student 权重，$\theta'$ 为 teacher/EMA 权重。

## 核心要点
1. 在 [[DINO]]、[[JEPA]] 中用于稳定 teacher 网络
2. 在扩散模型中用于生成质量更好的 EMA 权重推理
3. $\mu$ 越大，更新越慢、越稳定

## 相关概念
- [[DINO]]
- [[JEPA]]
- [[DiT]]
