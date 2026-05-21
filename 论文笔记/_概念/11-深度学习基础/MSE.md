---
type: concept
aliases: [均方误差, Mean Squared Error, L2 Loss]
---

# MSE

## 定义
**均方误差**：预测值与目标值之间差的平方的平均，是回归任务最常用的损失函数。

## 数学形式
$$
\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N}\|\hat{y}_i - y_i\|_2^2
$$

## 核心要点
1. 等价于在 Gaussian 似然下的最大似然估计
2. 对 outlier 敏感（平方放大大误差）
3. 在世界模型预测中是默认选择，但**只能学到多模态分布的条件均值**——这是 [[JEPA-WM]] 等确定性预测器的根本局限
4. 替代方案：L1、Huber、Diffusion / Flow Matching（多模态）

## 代表工作
- [[JEPA-WM]]: 嵌入空间 MSE 损失
- [[DINO-WM]]: 嵌入空间 MSE 损失

## 相关概念
- [[Flow Matching]]
- [[扩散变换器]]
- [[JEPA]]
