---
type: concept
aliases: [GMM, 高斯混合, 高斯混合模型, Gaussian Mixture Model]
---

# GMM

## 定义
高斯混合模型 (Gaussian Mixture Model) 用多个加权高斯分量的和来逼近任意复杂分布，是一种经典的概率密度建模工具。

## 数学形式

$$
p(x) = \sum_{m=1}^{M} w_m\,\mathcal{N}(x\mid\mu_m,\sigma^2_m),\qquad \sum_m w_m = 1
$$

## 核心要点
1. 每个分量由权重 $w_m$、均值 $\mu_m$、方差 $\sigma^2_m$ 描述。
2. 分量足够多时可逼近任意连续分布，常作为理论分析的工具分布。
3. 在 [[CoME]] 的理论分析中，每个记忆专家被建模为 GMM，用于推导[[Mixture of Contrastive Experts|对比专家]]只重加权权重、不改变核形状。

## 代表工作
- [[CoME]]: 用 GMM 假设证明 Proposition 1

## 相关概念
- [[核密度估计]]
- [[Mixture of Contrastive Experts]]
