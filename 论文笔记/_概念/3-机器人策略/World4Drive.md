---
type: concept
aliases: [World4Drive]
---

# World4Drive

## 定义

World4Drive 是基于 [[World Action Model|WAM]] 思想的端到端 [[自动驾驶]] 规划方法，属于"先 predict 再 plan"的顺序式 WAM 代表之一。在 nuScenes 上曾是 perception-free 端到端规划 SOTA 之一，被 [[DAWN]] 列为关键对比基线。

## 核心要点

1. **Sequential predict-then-plan**: 先预测未来场景，再据此规划，动作不会反向影响 world 预测
2. **多输入**: 通常 camera + LiDAR
3. **基准成绩**（[[DAWN]] 论文报告）:
   - NAVSIM v1 PDMS 85.1
   - nuScenes 平均 L2 0.50m / 碰撞率 0.16%

## 局限

- 动作条件在**冻结**的未来上，未来一旦错估动作无法纠正它
- 是 [[WAIM]] 范式想要超越的代表性"决策解耦"设计

## 相关概念

- [[World Action Model]]
- [[WAIM]]
- [[自动驾驶]]
- [[DAWN]]
