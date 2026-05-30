---
type: concept
aliases: [去噪损失, Denoising Loss]
---

# Denoising Loss

## 定义

[[Diffusion Model|扩散模型]] 训练时的核心损失：让模型预测加在 clean sample 上的噪声（或速度场），常用 MSE 形式。

## 数学形式

$$
\mathcal{L}_{\mathrm{denoise}}(\theta; x) = \mathbb{E}_{t, \epsilon}\!\left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
$$

其中 $x_t = \alpha_t x + \sigma_t \epsilon$，$t$ 为扩散时间步。

## 核心要点

1. 在 [[Flow Matching]] 中变体为速度预测损失 $\| v - v_\theta \|^2$
2. 可作为模型对样本"困惑度"的代理 — [[YoCausal]] 借此构造 [[Reverse Surprise Index|RSI]]
3. 对训练数据分布高度敏感，跨模型直接比较需要谨慎
4. 通常需对时间步 $t$ 做加权（如 [[EDM 噪声加权]]）

## 代表工作

- [[Diffusion Model]]、[[Flow Matching]]
- [[YoCausal]]: 把 denoising loss 当作世界知识探针的信号

## 相关概念

- [[Diffusion Model]]
- [[Flow Matching]]
- [[Score Function]]
