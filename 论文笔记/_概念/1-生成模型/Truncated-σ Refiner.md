---
type: concept
aliases: [Truncated-σ Refiner, Truncated Sigma Refiner, 截断 σ 精修器]
---

# Truncated-σ Refiner

## 定义

一种用于长视频两阶段生成的精修器训练范式：不从全噪声 $\sigma = 1$ 出发，而是从截断噪声水平 $\sigma_{\text{start}} < 1$ 开始做 [[Flow Matching]] 训练，只让精修器学习"短程修正"，避免破坏一阶段输出的长程结构。

## 数学形式

$$
x_{1} = (1 - \sigma_{\text{start}}) x_{\ell} + \sigma_{\text{start}} \epsilon, \quad
\alpha = \frac{\sigma_t}{\sigma_{\text{start}}}, \quad
x_t = (1 - \alpha) x_h + \alpha x_1, \quad
v^\star = \frac{x_1 - x_h}{\sigma_{\text{start}}}
$$

损失：

$$
\mathcal{L}_{\text{refiner}} = \mathbb{E}\, \|v_\theta(x_t, \sigma_t, c) - v^\star\|_2^2
$$

## 核心要点

1. 训练数据为配对 $(x_\ell, x_h)$，$x_\ell$ 为一阶段粗糙输出，$x_h$ 为高质量目标
2. 截断 $\sigma_{\text{start}}$（如 0.909375）决定精修强度，越小修正越弱、越保守
3. 通常配合 [[LoRA]] 训练，避免改动大基座权重
4. 用于长视频时显著优于直接套用原生短视频精修器（避免分布偏移）

## 代表工作

- [[SANA-WM]]: 用 $\sigma_{\text{start}} = 0.909375$ 在 17B LTX-2 上训 LoRA rank-384 做 60 秒视频精修

## 相关概念

- [[Flow Matching]]
- [[LoRA]]
- [[LDM]]
