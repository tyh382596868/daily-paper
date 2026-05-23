---
type: concept
aliases: [SmoothL1, Smooth L1 Loss, Huber Loss, 平滑 L1 损失]
---

# SmoothL1 Loss

## 定义

L1 与 L2 损失的混合形式：在残差较小时使用 L2（平滑、便于梯度），残差较大时切换到 L1（对 outlier 鲁棒）。等价于 Huber Loss 的特例（$\delta=1$）。

## 数学形式

$$
\mathrm{SmoothL1}(x) = \begin{cases}
0.5\, x^2 & |x| < 1 \\
|x| - 0.5 & |x| \geq 1
\end{cases}
$$

更一般的 Huber 形式（带阈值 $\delta$）：

$$
L_\delta(x) = \begin{cases}
0.5\, x^2 & |x| < \delta \\
\delta(|x| - 0.5\delta) & |x| \geq \delta
\end{cases}
$$

## 核心要点

1. **小残差用 L2**: 在原点附近连续可导，梯度随残差线性变化，训练稳定
2. **大残差用 L1**: 限制 outlier 的梯度幅值，避免单一离群样本主导更新
3. **vs. [[MSE]]**: MSE 对 outlier 敏感，可能让训练发散
4. **vs. L1**: L1 在 0 点不可导，梯度恒定可能导致小残差也"猛拽"
5. **典型应用**: 目标检测 bbox 回归（Fast R-CNN 原始用法）、深度回归、机器人深度/位姿监督

## 代表工作

- Fast R-CNN（Girshick, 2015）: 首次大规模用于 bbox 回归
- [[EvoScene-VLA]]: 用作跨视图深度重建的局部锚损失 $\mathcal{L}_{\text{geo}}$

## 相关公式

`torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)`：实际实现中 $\beta$ 控制阈值。

## 相关概念

- [[MSE]]
- [[扩散损失]]
- [[两级几何锚定]]
