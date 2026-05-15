---
type: concept
aliases: [LAW, Latent Action World Model]
---

# LAW (Latent Action World Model)

## 定义

LAW 是早期把 latent action 与 world model 联合训练的端到端 [[自动驾驶]] 方法，是 perception-free 规划研究的代表性早期工作之一。被 [[DAWN]] 列为基准对照。

## 核心要点

1. **Latent action**: 把动作表示在与 world latent 兼容的空间，便于联合训练
2. **属于 [[World Action Model|WAM]] 谱系**: 但未引入交互式互修
3. **基准成绩**（DAWN 报告）:
   - NAVSIM v1 PDMS 83.8
   - nuScenes 平均 L2 0.61m

## 局限

- world 与 action 的联合建模较表面，缺乏明确的耦合方向
- 性能在新方法（[[Drive-JEPA]]、[[DAWN]]）面前已显落后

## 相关概念

- [[World Action Model]]
- [[自动驾驶]]
- [[WAIM]]
