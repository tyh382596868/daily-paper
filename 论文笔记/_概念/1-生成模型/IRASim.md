---
type: concept
aliases: [IRA-Sim]
---

# IRASim

## 定义

一种动作条件的机器人视频生成基线方法，在 WorldArena Table 4 对比中 P3CScore 为 69.75。

## 核心要点

1. 以动作序列为条件控制视频生成
2. 在 WorldArena 基准上轨迹精度 0.364、交互质量 0.566
3. 与 CtrlWorld、Cosmos-Predict 等同属动作/轨迹条件生成基线

## 代表工作

- [[EA-WM]]: 在 Table 4 中与 IRASim 对比，验证 KVAF 表征的优越性
- [[RoboScape]]: 与 IRASim 对比视频生成质量及策略评估相关性，RoboScape 作为策略评估器的 Pearson 相关系数（0.953）远超 IRASim

## 相关概念

- [[世界模型]]
- [[KVAF]]
