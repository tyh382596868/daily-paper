---
type: concept
aliases: [RMBench, Robot Memory Bench]
---

# RMBench

## 定义

RMBench 是评估机器人策略**长程记忆能力**的仿真基准，包含 Battery Try、Blocks Ranking、Cover Blocks、Press Button 等任务，要求模型在执行中保留并利用任务历史信息。

## 核心要点

1. **长程记忆**: 任务步数长，需要记住前序子任务的结果
2. **典型任务**:
   - Battery Try（按顺序尝试电池）
   - Blocks Ranking（按顺序摆放方块）
   - Cover Blocks（覆盖指定方块）
   - Press Button（多步按按钮序列）
3. **对策略要求**: 显式记忆（如 [[Memory Bank]]）或隐式上下文记忆缺一不可

## 代表工作

- [[WLA]]: 56.5% 平均，刷新 SOTA（vs Fast-WAM 13.3%）
- [[DreamVLA]]: 30.3%
- Fast-WAM: 13.3%

## 相关概念

- [[Long-Horizon Manipulation]]
- [[Memory Bank]]
- [[LIBERO]]
