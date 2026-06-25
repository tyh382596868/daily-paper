---
type: concept
aliases: [Push-T, PushT Task]
---

# PushT

## 定义
经典 2D 机器人推物块（T 形）benchmark 任务，来自 Chi et al. (Diffusion Policy)，常用于评测扩散策略和世界模型的连续控制能力。

## 核心要点
1. 2D 连续控制：推动 T 形块对准目标位置
2. 任务本身简单但对精度和多模态策略要求高
3. 常用于 [[Diffusion Policy]]、[[DreamerV3]]、扩散蒸馏方法的 baseline 验证
4. 与 [[PushCube]]（3D ManiSkill 版本）形成对比

## 代表工作
- [[Diffusion Policy]]: PushT 作为主要演示任务
- [[ConformalWM]]: 用 PushT 验证世界模型可信域

## 相关概念
- [[PushCube]]
- [[Diffusion Policy]]
- [[ManiSkill]]
