---
type: concept
aliases: [PhyInf, 物理参数推断, Real-to-Sim 物理推断]
---

# Physics Inference

## 定义

OrbiSim 的 Real-to-Sim 子模块之一，从**多帧视频窗口**（64 帧）推断**隐藏物理参数**——密度、摩擦系数等无法从单帧看出的属性。

## 数学形式

$$
\bar{x}^{hidden} = f_\phi^{PhyInf}(o_{t-63:t})
$$

训练时通过冻结的 [[OrbiSim-Dynamics]] 做 rollout，用预测状态与真实状态的一致性 loss 反向监督：

$$
\mathcal{L}_{PhyInf} = \| \mathrm{rollout}_{dyn}(\bar{x}^{hidden}) - x_{0:T}^{gt} \|^2
$$

## 核心要点

1. **时序观测必要**: 摩擦/质量只能从运动差异中识别，单帧无法解算
2. **rollout-based 监督**: 利用可微 dynamics 做闭环训练
3. **冻结骨干**: 训练 PhyInf 时 OrbiSim-Dynamics / Vision 都冻结
4. **精度**: 物体位置 22 mm（含速度 28.63 mm/s）

## 代表工作

- [[OrbiSim]]: 提出 PhyInf 与 [[State Inference|StaInf]] 组成 Real-to-Sim pipeline

## 相关概念

- [[State Inference]]
- [[OrbiSim-Dynamics]]
- [[逆动力学]]
- [[基于模型的强化学习]]
