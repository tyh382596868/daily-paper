---
type: concept
aliases: [VICReg, Variance-Invariance-Covariance Regularization]
---

# VICReg

## 定义

VICReg (Bardes, Ponce & LeCun, ICLR 2022) 是一种**显式解耦**的自监督表示学习正则项，由三部分组成：
1. **Invariance**：不同视图嵌入间的 MSE
2. **Variance**：每个特征维度方差不低于阈值，防止坍塌到常数
3. **Covariance**：特征维度间协方差趋零，防止维度坍塌

无需对比负样本、EMA 或 stop-gradient。

## 数学形式

$$
\mathcal{L}_{\text{VICReg}} = \lambda \mathcal{L}_{\text{inv}} + \mu \mathcal{L}_{\text{var}} + \nu \mathcal{L}_{\text{cov}}
$$

其中：
- $\mathcal{L}_{\text{inv}} = \|\mathbf{z} - \mathbf{z}'\|^2$
- $\mathcal{L}_{\text{var}} = \sum_j \max(0, \gamma - \sqrt{\text{Var}(z_j)})$
- $\mathcal{L}_{\text{cov}} = \sum_{i \ne j} \text{Cov}(z_i, z_j)^2 / d$

## 核心要点

1. **显式控制方差/协方差**：相比对比学习无需负样本，相比 BYOL 无需 EMA
2. **多超参缺陷**：$\lambda, \mu, \nu, \gamma$ 加上对称版还需加几项，工程上调起来复杂
3. **被 [[PLDM]] 采用**：在世界模型场景被用作端到端训练时的反坍塌正则，但稳定性差

## 代表工作

- Bardes et al., ICLR 2022: 原始 VICReg
- [[PLDM]]: 把 VICReg 引入像素级世界模型
- [[LeWM]]: 用 [[SIGReg]] 替代 VICReg，超参从 6+ 降到 1

## 相关概念

- [[表征坍塌]]
- [[JEPA]]
- [[SIGReg]]
