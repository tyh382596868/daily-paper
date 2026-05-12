---
type: concept
aliases: [ManiSkill, ManiSkill2, ManiSkill3]
---

# ManiSkill

## 定义

ManiSkill 是一个基于 SAPIEN 物理引擎的**GPU 并行机器人操作仿真平台与 benchmark**，提供大量桌面操作任务、多种机器人本体、可微/高吞吐渲染，常用于训练与评测操作策略和[[世界模型]]。

## 核心要点

1. **GPU 向量化**：支持上百到上千个并行环境，适合大规模 RL / 世界模型 rollout。
2. **多本体多任务**：在 [[RLA-WM]] 论文中用到三个机器人（Panda、带 Robotiq 夹爪的 XArm、带圆柱末端的 UR10e）和五个任务（Pull Cube、Pull Cube with Tool、Roll Ball、Push T、Poke Cube），每任务约 1500 episode。
3. **跨本体评测**：同一任务可换不同机器人，便于测试策略/表征的跨本体泛化（[[残差潜在动作|RLA]] 的泛化实验即用此特性）。
4. **常见用途**：模仿学习、视觉 RL、世界模型预测质量评测。

## 代表工作

- ManiSkill / ManiSkill2 / ManiSkill3（Mu et al. 等）
- 作为主要评测平台出现在 [[RLA-WM]]

## 相关概念

- [[MuJoCo]]
- [[Push-T]]
- [[sim-to-real]]
- [[世界模型]]
