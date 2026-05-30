---
type: concept
aliases: [Model Predictive Path Integral, 路径积分模型预测]
---

# MPPI

## 定义

MPPI（Model Predictive Path Integral）是一种基于采样的随机最优控制算法，源自路径积分控制理论。它通过对动作扰动序列做加权平均（权重由 cost 通过 softmax 决定）来近似最优控制序列，相比 [[CEM]] 不丢弃低分样本而是全体加权。

## 数学形式

给定动作均值序列 $\bar{a}_{t:t+H}$，采样 $K$ 条扰动轨迹 $a^{(k)} = \bar{a} + \epsilon^{(k)}$，每条计算 cost $S^{(k)}$：

$$
\bar{a}^{\mathrm{new}}_{t:t+H} = \sum_{k=1}^{K} w^{(k)} a^{(k)}, \quad w^{(k)} = \frac{\exp(-\beta S^{(k)})}{\sum_{j} \exp(-\beta S^{(j)})}
$$

其中 $\beta$ 是温度系数。所有样本通过 softmax 加权融合，区别于 [[CEM]] 的 top-k elite 截断。

## 核心要点

1. **温度加权 vs 硬截断**：MPPI 用 softmax，[[CEM]] 用 top-k elite
2. **样本利用率高**：所有样本都参与更新，不丢弃信息
3. **连续控制友好**：在高维连续动作空间常优于 CEM
4. **可与 WM 组合**：在 [[世界模型]] 学到的 latent dynamics 上做 MPPI 是 model-based RL 经典路线

## 在 SWM 中的位置

[[StableWM]] 内置 MPPI 作为 [[CEM]] 之外的可选 planner，统一通过 `PlanConfig` 配置 horizon、receding horizon、warm-start 等参数。

## 代表工作

- [[StableWM]]: 内置 MPPI baseline planner
- Williams et al. 2017: MPPI 原论文 (Information Theoretic MPC)
- [[TD-MPC2]]: 在 latent space 用 MPPI 风格的规划

## 相关概念

- [[CEM]]
- [[MPC]]
- [[世界模型]]
