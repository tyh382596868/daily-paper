---
type: concept
aliases: [Smooth-L1 Loss, Huber Loss, Huber 损失, SmoothL1]
---

# Smooth-L1 Loss

## 定义

L1 与 L2 损失的组合，在残差 $|x| \le \beta$ 时表现为平方损失（梯度连续），在 $|x| > \beta$ 时退化为线性损失（对 outlier 鲁棒）。Faster R-CNN 中提出，常用于回归任务。

## 数学形式

$$
\ell_{\text{Huber}}(x; \beta) = \begin{cases}
\dfrac{1}{2\beta} x^2 & \text{if } |x| < \beta \\
|x| - \dfrac{\beta}{2} & \text{otherwise}
\end{cases}
$$

PyTorch 默认 $\beta = 1$。

## 核心要点

1. **Outlier 鲁棒**: 大残差用线性而非平方，梯度上界为 1
2. **小残差光滑**: 接近最优时梯度变小，收敛稳定
3. **目标缩放**: 监督目标若数值很大（如 [[TRM]] 里 $y_{ij}/s$, $s=224$），需先除以尺度让残差落在 $[-1, 1]$ 区间，损失更敏感
4. **与 L2 对比**: L2 对 outlier 敏感，会被异常大梯度主导；Smooth-L1 上限保证了稳定性

## 代表工作

- Girshick, 2015: Fast R-CNN 中提出
- [[TRM]]: 训练成对头使用 Smooth-L1 on scaled distances
- 大多数目标检测框架的回归头

## 相关概念

- [[L1 Loss]]
- [[L2 Loss]]
- [[TRM]]
