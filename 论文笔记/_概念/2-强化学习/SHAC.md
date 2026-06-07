---
type: concept
aliases: [SHAC, Short-Horizon Actor-Critic, 短视野 Actor-Critic]
---

# SHAC

## 定义

**Short-Horizon Actor-Critic (SHAC)** (Xu et al., ICLR 2022) 是一种在**可微分仿真器**上做策略优化的算法。区别于 [[PPO]] 这类无模型方法，SHAC 利用仿真器的解析梯度，沿短视野 rollout 反向传播策略梯度，配以 critic 提供长程价值估计。

## 数学形式

$$
\nabla_\theta J = \mathbb{E}\!\left[\sum_{t=0}^{H} \nabla_\theta \big(r_t + \gamma V_\phi(s_{t+H})\big)\right]
$$

通过可微仿真器，$\nabla_\theta r_t$ 可解析计算，无需 score-function estimator。

## 优势 / 局限

- 优势：低方差梯度、样本效率高、配合可微仿真器 (e.g. [[DiffAero]]) 训练快。
- 局限：需要仿真器可微；接触/刚体碰撞往往不可微；视野 $H$ 不能太长否则梯度爆炸/消失。

## 代表工作

- 原论文：Accelerated Policy Learning with Parallel Differentiable Simulation (ICLR 2022)
- [[MAD]]：作为下游策略之一（MAD-SHAC）

## 关联概念

- [[Actor-Critic]]
- [[PPO]]
- [[DiffAero]] / [[Physics Simulator|可微仿真]]
