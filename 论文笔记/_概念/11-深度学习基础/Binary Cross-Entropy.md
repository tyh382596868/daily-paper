---
type: concept
aliases: [BCE, Binary Cross-Entropy, 二元交叉熵]
---

# Binary Cross-Entropy（BCE）

## 定义

二分类任务的标准交叉熵损失：衡量模型预测概率 $p$ 与 ground truth 标签 $y \in \{0,1\}$ 之间的差异。

## 数学形式

$$
\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]
$$

其中 $p_i = \sigma(\hat{y}_i)$，$\sigma(\cdot)$ 为 sigmoid。

## 核心要点

1. **概率匹配**: 当 $y=1$ 时只惩罚 $\log p$，当 $y=0$ 时只惩罚 $\log(1-p)$。
2. **像素/区域级监督**: 常用于分割、affordance map 等密集预测任务。
3. **数值稳定**: 实践中用 BCEWithLogitsLoss 在 logit 上直接计算，避免溢出。
4. **类别不平衡**: 正负样本悬殊时需加权或用 Focal Loss 改进。

## 代表工作

- [[AffordanceVLA]] 的 [[Where2Act Loss]]
- 各类分割模型

## 相关概念

- [[Cross-Entropy]]
- [[Focal Loss]]
- [[Sigmoid]]
