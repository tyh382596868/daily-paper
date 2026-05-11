---
type: concept
aliases: [OGBench, Offline Goal-conditioned Benchmark]
---

# OGBench

## 定义

OGBench (Park et al., 2024) 是面向**离线目标条件强化学习**的 benchmark 套件，覆盖 navigation、locomotion、manipulation 等多类任务。提供统一的离线数据集 + 标准评估协议，支持目标条件 RL、世界模型、模仿学习的横向比较。

## 核心要点

1. **离线 + 目标条件**：所有数据预收集，agent 通过 latent 比较或 reward shaping 朝目标推进
2. **多类任务**：包括 OGBench-Cube（3D 机械臂操作）、PointMaze、AntMaze 等
3. **作为视觉世界模型评测**：OGBench-Cube 是 [[LeWM]] 与 [[DINO-WM]] 对比的关键 3D 视觉环境

## 代表工作

- Park et al., 2024: OGBench 原始论文
- [[LeWM]]: 在 OGBench-Cube 上略输 [[DINO-WM]]，作者归因 3D 视觉复杂度

## 相关概念

- [[Push-T]]
- [[世界模型]]
