---
type: concept
aliases: [Dreamer, Dreamer 算法族]
---

# Dreamer

## 定义

**Dreamer** 是 Hafner et al. 提出的基于[[潜空间世界模型]]的 [[Model-Based RL|model-based RL]] 算法族（Dreamer / DreamerV2 / [[DreamerV3]] / Dreamer 4 等），通过在 latent 中做 imagination rollout 训练 actor-critic，显著降低环境交互成本。

## 核心范式

1. **学习世界模型**：用 [[循环状态空间模型|RSSM]] 把观测压成 latent $(h_t, z_t)$，预测下一步 latent。
2. **在 latent 中想象**：从真实 trajectory 切片出发，模型自回归 rollout 一段 latent 序列，作为虚拟训练数据。
3. **在想象中训练 actor-critic**：用解码出的奖励/继续标志算 critic target，用解码出的 latent 反向传策略梯度。

## 演变线索

- **Dreamer / V2 / V3**：纯仿真任务（Atari、DMControl、Crafter 等）逐步扩大规模。
- **Dreamer 4** ([[Dreamer 4]])：被推到视觉复杂的开放环境。
- **[[MAD]]**：把 Dreamer 范式落地到四旋翼自主飞行——同时把视觉重建头换成 OGM/VGM。

## 关联概念

- [[DreamerV3]]
- [[Latent Dynamics Rollout|想象 rollout]]
- [[World Model]] / [[潜空间世界模型]]
- [[循环状态空间模型|RSSM]]
- [[MAD]]
