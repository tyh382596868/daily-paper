---
type: concept
aliases: [Soft Actor-Critic, 软演员评论家]
---

# SAC

## 定义
一种最大熵强化学习算法，在优化累积奖励的同时最大化策略熵，提升探索能力和鲁棒性。

## 数学形式
$$\mathcal{J}(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$
其中 $\alpha$ 为温度参数，$\mathcal{H}$ 为策略熵。

## 核心要点
1. off-policy 算法，样本效率高于 PPO
2. 自动调节温度参数 $\alpha$
3. 适合连续动作空间的机器人控制任务

## 代表工作
- Haarnoja et al., 2018: SAC 原始论文

## 相关概念
- [[PPO]]
- [[DreamerV3]]
