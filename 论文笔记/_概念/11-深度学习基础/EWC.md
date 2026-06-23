---
type: concept
aliases: [Elastic Weight Consolidation, 弹性权重巩固]
---

# EWC（Elastic Weight Consolidation）

## 定义
一种持续学习正则化方法，通过对旧任务重要权重施加弹性约束，在学习新任务时防止灾难性遗忘。

## 数学形式
$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_{A,i})^2$$

其中 $F_i$ 是 Fisher 信息矩阵的对角元素（衡量参数对旧任务的重要性），$\theta^*_A$ 是旧任务训练后的参数，$\lambda$ 控制约束强度。

## 核心要点
1. 用 Fisher 信息矩阵近似参数重要性
2. 对重要参数施加二次惩罚，防止其被新任务大幅更新
3. 计算开销低，无需保存旧数据（不同于 replay buffer 方法）
4. 适用于顺序任务学习场景（robot lifelong learning）

## 代表工作
- [[Kirkpatrick2017]]：EWC 原始论文（Overcoming catastrophic forgetting in neural networks）
- [[RECALL]]：在 VLA 持续学习中与主动经验收集结合使用

## 相关概念
- [[灾难性遗忘]]
- [[Continual Learning]]
- [[LoRA]]
