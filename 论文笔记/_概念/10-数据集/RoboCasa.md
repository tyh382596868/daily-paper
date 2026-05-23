---
type: concept
aliases: [RoboCasa]
---

# RoboCasa

## 定义
一个大规模家庭机器人操作仿真 benchmark，提供多样化的厨房/家庭场景和 100+ 个操作任务，用于训练和评估通用机器人策略。

## 核心要点
1. 基于 MuJoCo 物理引擎和 Robosuite 框架
2. 支持数据增强（场景布局随机化、物体替换）
3. 在 [[RLDX-1]]、[[OpenVLA]] 等论文中作为训练数据和评估 benchmark

## 代表工作
- [[GaussianDream]]: 在 Human-50 协议上达到平均 52.6%，Pick&Place 类任务 43.8% 显著领先

## 相关概念
- [[LIBERO]]
- [[SimplerEnv]]
- [[VLA]]
