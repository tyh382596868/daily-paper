---
type: concept
aliases: [OOV, Out-of-Vision, 视野外操作, Out of Vision]
---

# OOV (Out-of-Vision)

## 定义

机器人操作中的一类设置——任务相关物体**当前不在相机视野内**，策略必须依赖记忆 / 主动感知 / 跨视角推理才能定位目标。

## 核心要点

1. 现有 [[VLA]] 隐式假设目标可见，遇到 OOV 即失效，表现为反复摇头、试抓
2. OOV 任务类型：
   - **Invisible → Invisible**: 起点终点都不可见
   - **Visible → Invisible**: 起点可见，目标不可见
   - **Invisible → Visible**: 起点不可见，目标可见
   - **Sequential**: 多步连续 OOV
   - **Dual-Arm**: 双臂协调 OOV
3. 评测需要超越成功率的**行为指标**：注视时间、搜索路径、视角修正次数等
4. 解决思路：主动头扫描 + 空间记忆（[[SOMA]]）、长程时间记忆（[[PrediMem]]、[[MemoryVLA]]）

## 代表工作

- [[SOMA]]: 首次系统化 OOV PnP benchmark + 行为指标
- [[MemoryVLA]]、[[PrediMem]]: 时间记忆方向（与空间正交）

## 相关概念

- [[VLA]]
- [[空间记忆专家]]
- [[关键帧记忆库]]
- [[MemoryVLA]]
- [[PrediMem]]
