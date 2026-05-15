---
type: concept
aliases: [Crossing-Path 任务族, Crossing-Path tasks]
---

# Crossing-Path 任务族

## 定义

[[AliasBench]] 的第二类任务：在某些中间状态下机械臂处在**多条可能路径的交叉点**，下一步走哪条路完全由 episode 早期选择的源点 / 目标决定，仅凭当前帧无法区分。

## 核心要点

1. **典型例子**: Cook Bread（手持面包，路径分叉到"煎锅"或"盘子"）
2. **混叠源**: 路径在中点 / 交叉点上重合
3. **解法**: 历史信号帮助确定源点
4. **任务数量**: 3 个

## 代表工作

- [[IntentVLA]]: 在 Crossing-Path 上从 15.7% 提升到 74.7%（+59.0 pp），是收益最大的任务族

## 相关概念

- [[AliasBench]]
- [[观测混叠]]
