---
type: concept
aliases: [物理仿真器, Physics Engine]
---

# Physics Simulator

## 定义

**Physics Simulator**（物理仿真器）是显式建模刚体 / 软体 / 流体 / 接触力学的计算引擎，给定初始状态和动作输入，按物理规律积分出未来状态。在 Physical AI 中常作为生成式 [[World Model]] 的对照或互补：仿真器保证物理一致性，生成模型保证视觉真实度。

## 核心要点

1. **代表**: MuJoCo / Bullet / Isaac Gym / Genesis / [[OrbiSim]]
2. **vs 生成式世界模型**: 仿真器物理正确但视觉简陋，生成模型反之
3. **混合趋势**: 把仿真器作为可微 prior 注入生成模型，例如 [[Cosmos3]] 批判中提到的改进方向
4. **机器人学习核心工具**: 大量 sim-to-real 训练依赖物理仿真器

## 代表工作

- [[OrbiSim]]: NVIDIA 仿真平台
- Isaac Lab / Isaac Sim
- MuJoCo

## 关联

- [[World Action Model]]
- [[Sim-to-Real]]
- [[Action-Conditioned World Model]]
