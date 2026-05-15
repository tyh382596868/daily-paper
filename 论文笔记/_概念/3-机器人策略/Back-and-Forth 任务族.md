---
type: concept
aliases: [Back-and-Forth 任务族, Back-and-Forth tasks]
---

# Back-and-Forth 任务族

## 定义

[[AliasBench]] 的第一类任务：同一物体在两个或多个位置之间**反复来回**搬运，每次搬运中间存在视觉上几乎相同的"持物"状态，但下一步动作方向不同。

## 核心要点

1. **典型例子**: Move Phone（把手机来回挪动几次）
2. **混叠源**: 中间持物姿态在不同 round 都长一样
3. **解法**: 必须靠最近几帧的"我从哪儿来"信号
4. **任务数量**: [[AliasBench]] 中包含 4 个 Back-and-Forth 任务

## 代表工作

- [[IntentVLA]]: 在 Back-and-Forth 上从 6.0% 提升到 49.3%（+43.3 pp）

## 相关概念

- [[AliasBench]]
- [[观测混叠]]
