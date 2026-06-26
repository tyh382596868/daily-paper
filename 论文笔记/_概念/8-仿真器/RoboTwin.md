---
type: concept
aliases: [RoboTwin]
---

# RoboTwin

## 定义
基于 Isaac Sim 的双臂机器人操作仿真 benchmark，提供高质量渲染和多样化操作任务，支持 VLA 和世界模型的评估。

## 核心要点
1. 支持双臂机器人操作任务（StackBlock、HangMug 等）
2. 基于 NVIDIA Isaac Sim，光线追踪渲染
3. 常用于 WAM、VLA 类论文的仿真评估
4. 2.0 版本提供 31 个任务，含 Clean / Rand 两种位姿设置

## 代表工作
- [[EvoScene-VLA]]: 31 任务平均成功率 87.2 % → 89.1 %（Clean）/ 86.1 % → 88.5 %（Rand）
- [[ForesightSafety-VLA]]: 在 RoboTwin 上构建 66 个安全增强场景，覆盖 5 种机器人形态

## 相关概念
- [[MuJoCo]]
- [[SimplerEnv]]
- [[VLA]]
