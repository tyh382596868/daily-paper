---
type: concept
aliases: [Asymmetric Actor-Critic, 非对称 AC, Asymmetric AC]
---

# 非对称 Actor-Critic

## 定义
非对称 Actor-Critic（Asymmetric Actor-Critic）是一种 RL 训练框架，其中 Critic（价值函数）可以访问训练时才有的完整特权信息（privileged information），而 Actor（策略）只能访问部署时可获得的观测，使策略可以在现实中部署，同时在训练时享有丰富的状态信息。

## 数学形式

$$
V(s_{\text{priv}}) \approx \mathbb{E}\left[\sum_{t} \gamma^t r_t \mid s_{\text{priv}}\right]
$$

$$
\pi(a \mid o_{\text{actor}}) \quad \text{（Actor 只用部分观测）}
$$

## 核心要点
1. **Critic** 访问完整特权状态 $s_{\text{priv}}$（如完整物理状态、外力、地形信息等）
2. **Actor** 只访问可部署观测 $o_{\text{actor}}$（如本体感知、IMU、视觉等）
3. 训练时 Critic 为 Actor 提供更准确的价值估计，加速学习收敛
4. 部署时只运行 Actor，无需特权信息，可在真实硬件上运行
5. 广泛用于人形机器人、四足运动等 Sim-to-Real 任务

## 代表工作
- [[HANDOFF]]: 使用非对称 Actor-Critic 训练全身控制策略

## 相关概念
- [[Actor-Critic]]
- [[PPO]]
- [[强化学习]]
- [[Knowledge Distillation]]
