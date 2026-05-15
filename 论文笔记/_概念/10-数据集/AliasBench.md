---
type: concept
aliases: [AliasBench]
---

# AliasBench

## 定义

[[IntentVLA]] 提出的 12 任务机器人操作 benchmark，在 [[RoboTwin]] 2 上专门隔离**[[观测混叠]]**：同一画面 $o_t^{(1)} \approx o_t^{(2)}$ 却需要不同动作 $a_t^{(1)} \neq a_t^{(2)}$。

## 数学形式

$$
o_t^{(1)} \approx o_t^{(2)} \quad \wedge \quad a_t^{(1)} \neq a_t^{(2)}
$$

## 核心要点

1. **四类任务族**:
   - [[Back-and-Forth 任务族]]（4 任务）: 同一物体反复来回
   - [[Crossing-Path 任务族]]（3 任务）: 路径源点决定终点
   - [[Bimanual 任务族]]（2 任务）: 对称交接
   - [[Multi-Goal 任务族]]（3 任务）: 瞬时线索选目标
2. **每任务 100 条演示**，规模适中、专攻混叠
3. **配套诊断**: 用视觉嵌入做 top-5 retrieval，约 50% 邻居来自异意图轨迹，证明仅靠当前帧无法解
4. **配套指标 [[Inter-Chunk Consistency|ICC-L2]]**: 度量重规划带来的动作抖动

## 代表工作

- [[IntentVLA]]: 在 AliasBench 上把均值成功率从 9.0%（frame-only）推到 45.8%

## 相关概念

- [[观测混叠]]
- [[RoboTwin]]
- [[Inter-Chunk Consistency]]
