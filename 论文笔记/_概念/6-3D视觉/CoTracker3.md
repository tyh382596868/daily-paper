---
type: concept
aliases: [CoTracker3, CoTracker]
---

# CoTracker3

## 定义

Meta FAIR 提出的视频点追踪（point tracking）模型，sparse tracking 范式的代表。给定首帧查询点，输出该点在整段视频中的轨迹。CoTracker3 是 v3 版本，相比 RAFT 这种 dense flow，更适合做长时间一致的稀疏点追踪。

## 与 [[RAFT]] 的差异

| 维度 | RAFT | CoTracker3 |
|------|------|-----------|
| 范式 | dense optical flow | sparse point tracking |
| 时序长度 | 两帧 | 多帧序列 |
| 输出 | 全图位移场 | 选定点的轨迹 |
| 适合任务 | 像素级配准 | 物体跟随、长时一致性 |

## 在机器人中的应用

- 物体追踪驱动的策略学习
- [[AHEAD]] 中作为 RAFT 的消融对照（性能略低 ~10%）
- [[RoboScape]]: 用作关键点热图伪标签生成器，自动标注操作视频的物体动力学轨迹

## 相关概念

- [[RAFT]]
- [[光流]]
