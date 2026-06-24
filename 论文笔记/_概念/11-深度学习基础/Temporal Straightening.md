---
type: concept
aliases: [时序直线化, 轨迹直线化, Latent Trajectory Straightness]
---

# Temporal Straightening

## 定义

Temporal Straightening（时序直线化）描述表示空间中时序轨迹的线性程度：若相邻时刻的潜在速度方向高度一致（余弦相似性接近 1），则轨迹"直"，递归展开时误差累积更慢；若方向频繁反转（余弦相似性接近 -1），则轨迹"振荡"，递归展开不稳定。

## 数学形式

直线化指标（Straightness Score）：

$$
S_\mathrm{straight}^{(i)} = \frac{1}{T-2} \sum_{t=1}^{T-2} \frac{\langle \dot{\tilde{\mathbf{s}}}^{(i)}_t,\ \dot{\tilde{\mathbf{s}}}^{(i)}_{t+1} \rangle}{\|\dot{\tilde{\mathbf{s}}}^{(i)}_t\| \cdot \|\dot{\tilde{\mathbf{s}}}^{(i)}_{t+1}\|}
$$

其中 $\dot{\tilde{\mathbf{s}}}^{(i)}_t = \tilde{\mathbf{s}}^{(i)}_{t+1} - \tilde{\mathbf{s}}^{(i)}_t$ 为第 $i$ 条轨迹在时刻 $t$ 的潜在速度。

## 核心要点

1. 源自感知心理学：大脑表示空间中的时序轨迹趋向直线化，使预测更稳定
2. [[JEPA]] 风格的预测目标隐式促进直线化，而自回归重建目标往往产生振荡轨迹
3. 直线化轨迹使递归展开的误差累积率（Compounding Ratio）更低
4. 值域 $[-1, 1]$：接近 1 最佳（直线），接近 -1 最差（振荡）

## 代表工作

- [[SkyJEPA]]: 用直线化指标比较 JEPA、Predictive、Reconstruction 三种方法，JEPA 均值约 0.75，Predictive baseline 约 -0.4

## 相关概念

- [[JEPA]]
- [[World Model]]
- [[SIGReg]]
