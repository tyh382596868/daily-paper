---
type: concept
aliases: [W1, Earth Mover Distance (1-阶), EMD]
---

# Wasserstein-1 Distance

## 定义

最优运输框架下衡量两个概率分布差异的 1-阶 Wasserstein 距离，也称 Earth Mover's Distance。

## 数学形式

$$
W_1(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu,\nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|x-y\|\, \mathrm{d}\gamma(x,y)
$$

在一维下有闭式：$W_1(\mu, \nu) = \int_0^1 |F^{-1}_\mu(u) - F^{-1}_\nu(u)|\,\mathrm{d}u$。

## 核心要点

1. 比 KL / JS 更适合不重叠分布，给出连续梯度
2. 在 GAN（WGAN）、轨迹动力学度量中常用
3. [[Dream-exe]] 用其度量"逐帧速度分布"差异（DYN）

## 代表工作
- [[Dream-exe]]：DYN 指标
- WGAN、Sinkhorn 系列

## 相关概念
- [[Optimal Transport]]
- [[Symmetric Hausdorff Distance]]
- [[Dynamic Time Warping]]
