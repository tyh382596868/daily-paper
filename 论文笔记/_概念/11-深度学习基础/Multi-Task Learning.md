---
type: concept
aliases: [多任务学习, MTL, 联合训练]
---

# Multi-Task Learning

## 定义

多任务学习（Multi-Task Learning, MTL）是通过同时优化多个相关任务的损失函数来训练共享表示的机器学习范式，旨在利用任务间的共享信息提升各任务的泛化性能。

## 数学形式

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{T} \lambda_i \mathcal{L}_i(\theta_{\text{shared}}, \theta_i)
$$

其中 $T$ 为任务数，$\lambda_i$ 为各任务权重，$\theta_{\text{shared}}$ 为共享参数。

## 核心要点

1. **正则化效果**: 多任务监督信号防止过拟合单一任务，提升共享表示的通用性
2. **辅助任务设计**: 辅助任务不必是最终目标，但需与主任务共享有价值的中间表示
3. **损失权重调节**: 各任务权重 $\lambda_i$ 需要仔细调节，防止某任务主导训练
4. **渐进式引入**: 复杂辅助损失可以在训练早期预热引入（如 OA-WAM 的 $\lambda_c$ 预热策略）

## 代表工作

- [[OA-WAM]]: 主任务动作预测 + 世界预测/图像重建/结构一致性/角色分类四个辅助损失

## 相关概念

- [[Flow Matching]]
- [[World Model]]
- [[VLA]]
