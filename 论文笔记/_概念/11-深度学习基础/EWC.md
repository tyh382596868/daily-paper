---
type: concept
aliases: [EWC, Elastic Weight Consolidation, 弹性权重巩固]
---

# EWC

## 定义
持续学习（Continual Learning）中防止灾难性遗忘的经典方法，通过在损失函数中加入基于 Fisher 信息矩阵的权重正则项，约束新任务训练时对旧任务重要权重的修改幅度。

## 数学形式
$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_{A,i})^2$$

其中 $F_i$ 为任务 A 的 Fisher 信息（参数重要性估计），$\theta^*_{A}$ 为任务 A 学习后的参数。

## 核心要点
1. Fisher 信息矩阵衡量参数对旧任务的重要程度（曲率近似）
2. 弹性约束：重要参数不能偏离旧值太多，不重要参数可自由调整
3. 计算开销：需存储旧任务的 Fisher 矩阵，内存随任务数线性增长

## 代表工作
- [[EWC]]: Kirkpatrick et al., 2017 (DeepMind)，提出原始方法
- [[RECALL-VLA]]: 在 VLA 终身学习中将 EWC 作为竞品基线

## 相关概念
- [[Continual Learning]] — 方向概念
- [[灾难性遗忘]] — 解决的核心问题
- [[Knowledge Distillation]] — 另一类防遗忘方法
