---
type: concept
aliases: [Recurrent State Space Model, 循环状态空间模型, Latent Dynamics Model]
---

# RSSM

## 定义
RSSM（Recurrent State Space Model）是 DreamerV1/V2/V3 系列中用于 latent 世界模型的核心架构，将潜状态分解为确定性（GRU）和随机性（VAE）两部分，支持纯潜空间内的多步想象（imagination）。

## 数学形式
潜状态 $h_t$（确定性）+ $z_t$（随机性）：

$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(GRU transition)}$$

$$z_t \sim q_\phi(z_t | h_t, o_t) \quad \text{(posterior)}$$

$$\hat{z}_t \sim p_\phi(\hat{z}_t | h_t) \quad \text{(prior, for imagination)}$$

训练损失 = 重建损失 + KL 散度（posterior vs prior）。

## 核心要点
1. 确定性 GRU 捕捉长程依赖，随机 VAE 捕捉环境随机性
2. 推理时用 prior 纯潜空间展开，无需观测，实现高效想象
3. 被 DreamerV1/2/3 广泛使用，是 MBRL 最经典的潜在动力学模型
4. SWAP 在 RSSM 上加了对称性约束（SymLoss）

## 代表工作
- [[DreamerV3]]: 最新版 RSSM，scalar reward + 自适应 KL
- [[SWAP]]: RSSM + 左右对称等变约束，用于足式跑酷
- [[DreamerV2]]: KL balancing + categorical latents

## 相关概念
- [[DreamerV3]]
- [[MBRL]]
- [[World Model]]
- [[VAE]]
