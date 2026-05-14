---
type: concept
aliases: [UniAD, Unified Autonomous Driving]
---

# UniAD

## 定义

UniAD 是 2023 CVPR Best Paper 提出的端到端 [[自动驾驶]] 统一框架，将检测、跟踪、建图、运动预测、占据预测与规划串联在一个共享 Transformer 网络中，是 perception-based 端到端规划的代表性基线。

## 核心要点

1. **全任务统一**: 一个 backbone 同时输出感知/预测/规划，所有任务共享 query
2. **Query-centric**: 用 agent query / map query / motion query 串联
3. **基准成绩**（nuScenes）:
   - 平均 L2 1.03m / 碰撞率 0.31%
4. **依赖完整感知监督**: 与 perception-free 方法形成范式对比

## 与 perception-free 方法的关系

UniAD 的强感知监督在小规模数据上稳定有效，但对大规模无标签视频泛化弱；perception-free 方法（[[Drive-JEPA]] / [[DAWN]]）则通过自监督表征绕过这一限制。

## 相关概念

- [[自动驾驶]]
- [[Transformer]]
- [[DAWN]]
