---
type: concept
aliases: [World Eval]
---

# WorldEval

## 定义

把[[世界模型]]作为**离线策略评估器** — 完全在想象中对不同机器人策略做排序，避免真实部署成本。

## 核心要点

1. [[RobotWM-Survey]] Section 4.2 "policy evaluation" 类别的代表
2. 与 WorldArena 同属"以 WM 替代真机评估"的路线
3. 评估有效性取决于 WM 的物理一致性与 action-fidelity，hallucination 会污染评估信号

## 代表工作

- Li et al., 2025e: WorldEval 原始论文

## 相关概念

- [[世界模型]]
- [[GPC]]
- [[IRASim]]
- [[CtrlWorld|Ctrl-World]]
- [[RobotWM-Survey]]
