---
type: concept
aliases: [Dreamer V3, DreamerV3]
---

# DreamerV3

## 定义
基于 Recurrent State Space Model（RSSM）的世界模型，通过在 latent 空间中想象未来轨迹来训练策略，无需与真实环境大量交互。

## 数学形式
$$\text{RSSM}: \quad h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim q(z_t | h_t, x_t)$$
$$\mathcal{L} = \mathbb{E}_{q}[\log p(x_t|h_t,z_t) + \log p(r_t|h_t,z_t)] - \beta \cdot \text{KL}[q \| p]$$

## 核心要点
1. 统一处理离散和连续动作空间，单套超参数适用多个 domain
2. 引入 symlog 变换处理大幅度值范围的 reward
3. 在 Atari、DMC、Crafter 等 benchmark 上 SOTA

## 代表工作
- Hafner et al., 2023: DreamerV3 原始论文

## 相关概念
- [[JEPA]]
- [[HaM-World]]
- [[CEM]]
