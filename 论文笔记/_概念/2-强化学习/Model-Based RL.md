---
type: concept
aliases: [Model-Based RL, MBRL, 基于模型的强化学习]
---

# Model-Based RL

## 定义

**Model-Based RL (MBRL)** 指**学习或使用环境动力学模型**辅助策略训练或决策的一类强化学习方法，与 [[PPO]] / SAC 等 [[Model-Free RL|model-free]] 方法对照。

## 与 Model-Free 的核心区别

| 维度 | Model-Free | Model-Based |
|------|------------|-------------|
| 是否学环境模型 | 否 | 是 |
| 样本效率 | 低 | 高 |
| 渐近性能 | 高 | 受模型偏差限制 |
| 计算开销 | 低 | 高（模型 + 想象 rollout） |

## 主要范式

1. **背景规划 (Background planning)**：用 [[Dyna]]、[[Dreamer]] 等在学到的模型里 rollout，生成想象数据补充真实经验。
2. **决策时规划 (Decision-time planning)**：[[MPC]]、[[MPPI]]、[[MuZero]]、[[TD-MPC]]——每个时刻用模型做 N 步前瞻规划再决策。
3. **可微仿真梯度**：[[SHAC]]、可微物理——直接对解析仿真器梯度做策略优化。
4. **[[潜空间世界模型]]**：在 latent 空间建模，避免像素空间复合误差，代表 [[DreamerV3]] / [[TD-MPC2]] / [[MAD]]。

## 关键挑战

- [[Compounding Errors|长程复合误差]]：模型小误差在 rollout 中放大。
- 模型 vs 策略性能 trade-off：模型过拟合在分布外失败。
- 探索：通常需要额外信号（curiosity、reward-free pretraining）。

## 代表工作

- [[Dreamer]] / [[DreamerV3]] / [[Dreamer 4]]
- [[TD-MPC]] / [[TD-MPC2]]
- [[MuZero]]
- [[MAD]]
- [[PETS]]

## 关联概念

- [[World Model]]
- [[Latent Dynamics Rollout]]
- [[Compounding Errors]]
- [[PPO]]（Model-Free 对照）
