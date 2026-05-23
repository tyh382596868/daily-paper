---
type: concept
aliases: [TwoRoom, Two-Room, Two Room, 双房间]
---

# TwoRoom

## 定义

一个 2D 导航 benchmark：一个 agent 在被墙隔开的两个房间里，目标可能在同房间或跨墙房间，必须**绕过墙上的门洞**才能到达。是评估 [[Latent MPC]] 的经典 sanity check 环境，几乎所有 latent world model 论文 ([[LeWM]]、[[PLDM]]、[[TRM]]) 都报 TwoRoom 结果。

## 数学形式

- **状态**: agent (x, y)
- **动作**: 连续 2D 速度
- **环境约束**: 墙 + 门洞 → 欧氏短路径常被阻断
- **成功**: 到达目标 (x, y) 容忍半径内

## 核心要点

1. **拓扑陷阱**: Euclidean 短路径常常**被墙挡住**——直接用欧氏距离做规划代价会失败
2. **内在维度低**: 2D 平面让 latent WM 容易学，但欧氏 latent MSE 仍可能失败 → 暴露**度量**而非**表征**问题
3. **多种 manifest**:
   - **balanced n40 / b50 / b150**: 20 same-room + 20 cross-wall
   - **matched n40**: 控制 Euclidean 距离的 cross-wall 子集
   - **hard n100** ([[TRM]] 主战场): 50 same-room + 50 cross-wall **all in high-distance bucket**, 47/50 cross-wall 必须过门洞
4. **典型成绩 (hard n100 三种子均值)**:
   - LeWM raw latent MSE: 7.0%
   - LeWM + [[TRM]]: 97.0%
   - PLDM raw latent MSE: 32.7%
   - PLDM + [[TRM]]: 84.0%
   - Oracle task-state cost: 100.0%（说明任务可行）

## 代表工作

- [[LeWM]]: 把 TwoRoom 作为简单 sanity check（且暴露了高斯先验 vs 低本征维度的不匹配）
- [[PLDM]]: 原始端到端 JEPA 用 TwoRoom 之一作 benchmark
- [[TRM]]: 用 hard n100 manifest 极大放大欧氏代价的失败模式，并用 TRM 修复

## 相关概念

- [[Latent MPC]]
- [[Terminal Proximity Cost]]
- [[World Model]]
