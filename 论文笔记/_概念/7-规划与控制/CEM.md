---
type: concept
aliases: [Cross-Entropy Method, 交叉熵方法]
---

# CEM

## 定义
一种基于采样的随机优化方法，通过迭代地从精英样本中拟合分布来逐步接近最优解，常用于世界模型规划。

## 数学形式
$$p^{(k+1)}(\theta) = \arg\min_{p} -\mathbb{E}_{p^{(k)}}[\mathbf{1}[f(\theta) \geq f^{(k)}_{\text{elite}}] \log p(\theta)]$$

## 核心要点
1. 迭代过程：采样 → 评估 → 选精英 → 拟合分布
2. 假设分布通常为高斯，精英样本比例通常取 10-20%
3. 在 [[DreamerV3]]、[[LeWorldModel]] 等世界模型中用于规划

## 相关概念
- [[MPC]]
- [[DreamerV3]]
- [[JEPA]]
