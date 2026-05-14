---
type: concept
aliases: [NAVSIM, NAVSIM v1, NAVSIM v2]
---

# NAVSIM

## 定义

NAVSIM 是基于 [[nuScenes]] 与 OpenScene 构建的**仿真闭环规则评估**基准，用一组确定性规则与轻量仿真器评估[[自动驾驶]]规划器，避免传统 open-loop L2 误差对真实轨迹过拟合的问题。

## 核心指标（v1）

- **NC (No-at-fault Collision)**: 无责碰撞率，越高越好
- **DAC (Drivable Area Compliance)**: 可行驶区域合规
- **EP (Ego Progress)**: 自车前进效率
- **C (Comfort)**: 舒适度（加速度抖动）
- **TTC (Time-to-Collision)**: 与最近碰撞物的时间余量
- **PDMS**: 上述五项的加权聚合指标（越高越好）

## v2 新增指标

- DDC（Driving Direction Compliance）
- TL（Traffic Light Compliance）
- LK（Lane Keeping）
- HC（History Comfort）
- EC（Extended Comfort）
- 聚合为 **EPDMS**

## 代表成绩（NAVSIM v1 perception-free 区）

- LAW: 83.8
- World4Drive: 85.1
- Epona: 86.2
- Drive-JEPA: 89.0
- **[[DAWN]]: 89.1**（perception-free SOTA）

## 相关概念

- [[nuScenes]]
- [[自动驾驶]]
- [[DAWN]]
- [[Drive-JEPA]]
