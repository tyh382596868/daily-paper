---
type: concept
aliases: [World Model Policy Optimization]
---

# WMPO

## 定义

强调**像素空间想象 + on-policy [[GRPO]]** 的世界模型强化学习方法，是 RL 训练 VLA 路线的代表。

## 核心要点

1. [[RobotWM-Survey]] Section 4.1 中"WM as RL simulator"的代表
2. 直接在像素级想象 rollout 上做 on-policy 优化（区别于 [[DreamerV3]] 的潜空间）
3. 使用 [[GRPO]] 作为优化算法，处理稀疏奖励
4. 与 World4RL、World-Gymnast、RehearseVLA 同属一系列

## 代表工作

- Zhu et al., 2026: WMPO 原始论文

## 相关概念

- [[世界模型]]
- [[GRPO]]
- [[强化学习]]
- [[UniSim]]
- [[RobotWM-Survey]]
