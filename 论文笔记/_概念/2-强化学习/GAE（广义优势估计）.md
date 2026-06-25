---
type: concept
aliases: [GAE, Generalized Advantage Estimation, 广义优势估计]
---

# GAE（广义优势估计）

## 定义

由 Schulman et al. (2016) 提出，通过 λ 加权的 TD 误差折扣和来估计策略优势，在方差和偏差之间取得平衡。

## 数学形式

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中单步 TD 误差为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## 核心要点

1. $\lambda = 0$：退化为 TD(0) 优势估计，低方差高偏差。
2. $\lambda = 1$：退化为 Monte Carlo 优势估计，高方差低偏差。
3. 实践中常用 $\lambda = 0.95$，$\gamma = 0.99$。
4. GAE 已成为现代 on-policy RL（PPO 等）的标准优势估计方法。

## 代表工作

- [[PPO]]: 默认使用 GAE 做优势估计
- [[ROAD-VLA]]: 在线 critic 使用 GAE 估计 $\hat{A}_t^{int}$

## 相关概念

- [[优势函数（Advantage Function）]]
- [[PPO]]
- [[强化学习]]
