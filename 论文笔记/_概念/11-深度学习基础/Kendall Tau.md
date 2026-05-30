---
type: concept
aliases: [Kendall τ, Kendall Tau, Kendall 相关系数]
---

# Kendall Tau

## 定义

衡量两个排序变量之间一致性的非参数相关系数。常用于评测指标排名的对齐度。

## 数学形式

$$
\tau = \frac{\#\text{Concordant} - \#\text{Discordant}}{\binom{n}{2}}
$$

取值 $[-1, 1]$。1 = 完全一致；0 = 无关；−1 = 完全相反。

## 核心要点

1. 比 Pearson 更鲁棒，不依赖正态性假设
2. 比 Spearman 对小样本更稳健，但计算复杂度 $O(n^2)$
3. 在评测论文中常用于"我的新指标与人类偏好/经典指标一致吗？"
4. [[YoCausal]] 用 Kendall $\tau$ 报告与人类偏好、模型规模、美学等指标的一致性

## 代表工作

- 经典统计学方法（Kendall 1938）
- [[YoCausal]]: 与美学 $\tau=0$ 是其评测正交性的关键证据

## 相关概念

- [[Reverse Surprise Index]]
- [[Causality Cognition Index]]
