---
type: concept
aliases: [Categorical Latent, 离散潜变量, Categorical 潜变量]
---

# Categorical Latent

## 定义

**Categorical Latent** 指把潜变量 $z$ 建模为多个独立 categorical 分布的乘积（即一组 one-hot 离散变量），常用形式是 $z \in \{0,1\}^{C \times K}$（$C$ 个 categorical，每个 $K$ 类）。在世界模型中由 [[DreamerV2]] 引入并被 [[DreamerV3]] / [[MAD]] 沿用。

## 数学形式

$$
z = [z^{(1)}, \ldots, z^{(C)}],\quad z^{(c)} \sim \mathrm{Categorical}(\pi^{(c)}),\quad \pi^{(c)} \in \Delta^{K-1}
$$

通过 [[straight-through estimator|Straight-Through]] 或 Gumbel-Softmax 反向传播梯度。

## 为什么用 categorical 而不是 Gaussian

- 离散性带来 **长程稳定**：连续 Gaussian latent 在 16+ 步想象 rollout 中容易漂移；离散 latent 自然限制状态空间。
- 与重建目标天然对齐：图像/地图等观测的分布是高度多模的，连续高斯难以表达。
- 实践证据：DreamerV2/V3 在 Atari / DMC 大幅改进的关键因素之一。

## 关联概念

- [[DreamerV3]]
- [[循环状态空间模型|RSSM]]
- [[straight-through estimator|Straight-Through]]
- [[KL 散度]]
