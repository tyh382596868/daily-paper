---
type: concept
aliases: [YOPO, You Only Plan Once]
---

# YOPO

## 定义

**You Only Plan Once (YOPO)** 是面向四旋翼端到端导航的学习方法：用一次前向推理直接从感知到飞行命令，绕过经典 mapping-planning-tracking 模块化栈，但保留了类似 [[Latent Dynamics Rollout|latent dynamics]] 的内部结构。

## 在 [[MAD]] 中的角色

- 作为主要**端到端学习基线**之一，与 EGO-Planner（经典）对照。
- 在 [[Gazebo]] 圆柱森林评测中：稠密场景下 YOPO 失败、稀疏场景下也比 MAD-Dreamer 慢得多。
- 室内走廊 5×40 m：完成 16.02 s / 峰速 3.71 m/s（MAD-Dreamer 8.36 s / 6.37 m/s）。

## 关联概念

- [[Quadrotor]] / 四旋翼导航
- [[Latent Dynamics Rollout]]
- EGO-Planner（经典基线）
