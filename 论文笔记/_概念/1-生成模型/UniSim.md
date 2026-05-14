---
type: concept
aliases: [Universal Simulator]
---

# UniSim

## 定义

把动作条件视频模拟与奖励生成耦合在一起的早期"世界模型作为模拟器"工作，是[[强化学习]] in [[世界模型]] 路线的代表。

## 核心要点

1. [[RobotWM-Survey]] Section 4.1 中"WM as RL simulator"的开创性工作
2. 视频生成与奖励预测共享 backbone，可直接在想象 rollout 上做策略训练
3. 启发了后续 World-Env、VLA-RFT、[[WMPO]] 等系列工作

## 代表工作

- Yang et al., 2024a: UniSim 原始论文

## 相关概念

- [[世界模型]]
- [[Controllable 世界模型]]
- [[WMPO]]
- [[强化学习]]
- [[RobotWM-Survey]]
