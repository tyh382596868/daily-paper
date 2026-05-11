---
type: concept
aliases: [TD-MPC, TD-MPC2, Temporal Difference Model Predictive Control]
---

# TD-MPC

## 定义

TD-MPC (Hansen et al., 2022) 与 TD-MPC2 (2024) 是一类 **model-based RL** 方法：在 latent 空间联合学习动力学模型、奖励模型与价值函数，用 [[模型预测控制|MPC]] + [[CEM]] 在线规划，并用 TD 学习的价值作为终端价值替代纯 rollout，从而支持长 horizon。需要奖励信号或特权状态访问。

## 数学形式

规划目标：

$$
\mathbf{a}^*_{1:H} = \arg\max_{\mathbf{a}_{1:H}} \sum_{t=1}^{H} \gamma^{t-1} r(\hat{\mathbf{z}}_t, \mathbf{a}_t) + \gamma^H V_\psi(\hat{\mathbf{z}}_{H+1})
$$

总损失 = 动力学预测损失 + 奖励预测损失 + TD 价值损失 + （可选）latent 一致性正则。

## 核心要点

1. **价值 + 模型双轨**：弥补纯 rollout 在长 horizon 的误差累积
2. **任务特定**：需要奖励，无法在纯离线无奖励数据上训练
3. **是 LeWM 的对比基线之一**：定位为 "Task-specific" 类世界模型

## 代表工作

- Hansen et al., 2022: TD-MPC
- TD-MPC2 (2024): 更强 multi-task 版本
- [[LeWM]]: 作为 task-specific 基线对比

## 相关概念

- [[模型预测控制|MPC]]
- [[CEM]]
- [[DreamerV3]]
- [[世界模型]]
