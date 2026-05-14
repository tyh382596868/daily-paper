---
type: concept
aliases: [Generative Predictive Control]
---

# GPC

## 定义

把[[世界模型]]作为**决策时评估器**的代表方法 — 在执行前用 WM 做预测性 look-ahead，对候选动作做排序和精修。

## 核心要点

1. [[RobotWM-Survey]] Section 4.2 中"rollout-based candidate assessment"的代表
2. 类似 [[MPC|模型预测控制]] 但用学习式 WM 替代物理模型
3. 与 [[IRASim]]、World-in-World、DreamPlan 同属在线评估路线
4. 评估器质量取决于 WM 的 action-fidelity

## 代表工作

- Qi et al., 2026: GPC 原始论文

## 相关概念

- [[世界模型]]
- [[MPC]]
- [[IRASim]]
- [[CtrlWorld|Ctrl-World]]
- [[RobotWM-Survey]]
