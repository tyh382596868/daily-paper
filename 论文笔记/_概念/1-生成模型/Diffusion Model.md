---
type: concept
aliases: [Diffusion Model, 扩散模型, DDPM, Diffusion]
---

# Diffusion Model

## 定义
扩散模型。通过逐步加噪声把数据破坏成高斯噪声、再训练一个网络逐步去噪还原，作为生成式建模的核心范式。

## 数学形式
**前向过程**（加噪）：
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
$$

**反向过程**（学习去噪）：
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

## 核心要点
1. **训练目标**：预测噪声 $\epsilon$（DDPM）、x₀、或 score $\nabla_x \log p(x)$（score-based）
2. **采样**：从 $x_T \sim \mathcal{N}(0, I)$ 出发，逐步去噪 $T \to 0$
3. **加速变种**：[[DDIM]]、DPM-Solver、Consistency Model、[[DMD]] 等

## 在 robotics / WM 中的应用
- [[Diffusion Policy]]：用扩散模型直接生成机器人动作序列
- [[CogVideoX]] / [[AnimateDiff]]：视频扩散，是 video world model 的基础
- [[minWM]]：把多步双向扩散蒸馏为少步 AR 视频生成器
- [[CGPO]]：用 critic 引导扩散采样做 RL

## 详细资料
参见 [[扩散模型]]（完整介绍）

## 相关概念
- [[Diffusion Policy]]、[[DDIM]]、[[DMD]]、[[DiT]]、[[扩散模型]]
