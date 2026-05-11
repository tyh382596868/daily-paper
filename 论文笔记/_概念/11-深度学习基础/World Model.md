---
type: concept
aliases: [World Model, 世界模型, 环境模型]
---

# World Model

## 定义

World Model 是一类学习环境动态的生成模型，通过对当前状态 $s_t$ 和动作 $a_t$ 建模，预测未来状态 $\hat{s}_{t+1}$，使智能体无需在真实环境中大量采样即可进行规划和策略优化。

## 数学形式

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

推广到视频预测：

$$
\hat{\mathbf{I}}_{t+1} = f_\theta(\mathbf{I}_{\leq t}, \mathbf{a}_{\leq t})
$$

## 核心要点

1. **状态空间表示**: 可在像素空间（pixel-level）或潜在空间（latent-level）建模
2. **规划辅助**: 利用预测的未来状态做 rollout，辅助 MBRL（基于模型的强化学习）
3. **生成式预训练**: 在大规模视频数据上预训练，迁移到机器人控制场景
4. **与策略结合**: 与动作头联合训练，形成 World Action Model 框架

## 代表工作

- [[World Action Model|OA-WAM]]: 将世界模型应用于对象级鲁棒机器人操作
- [[Cosmos-Policy]]: 基于 Cosmos 视频世界模型驱动的机器人策略
- [[LeWM]]: 端到端 JEPA 潜空间世界模型，单卡可训

## 相关概念

- [[World Action Model]]
- [[Flow Matching]]
- [[VLA]]
