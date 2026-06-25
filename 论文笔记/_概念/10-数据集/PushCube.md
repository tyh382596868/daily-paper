---
type: concept
aliases: [Push Cube Task]
---

# PushCube

## 定义
ManiSkill 仿真环境中的标准推方块任务，机械臂需将方块推到目标位置，常用于评测 VLA RL 微调和操作策略方法。

## 核心要点
1. 基于 [[ManiSkill]] 物理仿真器
2. 任务简单但评测方便，是 VLA RL fine-tuning 的标准 baseline 任务之一
3. 与 PickCube、StackCube 等共同构成 ManiSkill 基础任务集
4. FORCE、ConRFT 等 VLA RL 工作的主要验证任务

## 代表工作
- [[FORCE]]: 用 PushCube 验证 VLA RL 微调效率

## 相关概念
- [[ManiSkill]]
- [[PushT]]
- [[PickCube]]
