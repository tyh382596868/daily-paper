---
type: concept
aliases: [TD-MPC v2]
---

# TD-MPC2

## 定义

[[TD-MPC]] 的扩展版本，仍然把[[世界模型]]作为评估器，在**潜空间做梯度式规划**，是[[MPC|模型预测控制]] + 学习式 WM 路线的代表。

## 核心要点

1. [[RobotWM-Survey]] Section 4.2 "MPC" 类别的代表
2. 跨多个 domain 用单套超参数训练（继承 TD-MPC 的设计哲学）
3. 与 LeWorldModel 同属"端到端潜空间 WM + 规划"路线
4. 比像素级 WM 推理快很多，适合实时控制

## 代表工作

- Hansen et al., 2024: TD-MPC2 原始论文

## 相关概念

- [[TD-MPC]]
- [[MPC]]
- [[世界模型]]
- [[LeWM|LeWorldModel]]
- [[RobotWM-Survey]]
