---
type: concept
aliases: [CQL, Conservative Q-Learning, 保守 Q 学习]
---

# CQL（Conservative Q-Learning）

## 定义
一种离线强化学习算法，通过在 Q 函数学习中加入"压低分布外（OOD）动作 Q 值、抬高数据集内动作 Q 值"的保守正则项，缓解离线 RL 中常见的价值高估问题。

## 数学形式
在标准 Bellman 误差之外加一个保守正则项（直观形式）：

$$\min_{Q}\ \alpha\Big(\mathbb{E}_{s\sim\mathcal{D},a\sim\mu}[Q(s,a)] - \mathbb{E}_{s,a\sim\mathcal{D}}[Q(s,a)]\Big) + \tfrac12\,\mathbb{E}_{\mathcal{D}}\big[(Q - \hat{\mathcal{B}}^\pi\hat{Q})^2\big]$$

其中 $\mu$ 是会高估的"对抗"动作分布，$\mathcal{D}$ 是离线数据集。该项保证学到的 $Q$ 是真实值的下界（保守估计）。

## 核心要点
1. 离线 RL 的核心难点是"外推误差"——对数据集没覆盖的 (s,a) 估值不准且通常偏高，bootstrapping 会放大它。
2. CQL 不显式约束策略，而是直接把 OOD 动作的 Q 值"压下去"，使 argmax 自然落在数据支撑内。
3. 本质上相当于一个隐式的行为克隆 anchor——数据次优时会限制后续在线提升（[[RankQ]] 等工作正是针对这一点改进）。

## 代表工作
- Kumar et al., 2020：CQL 原始论文。
- [[RankQ]]：批评 CQL 式"统一压低 OOD 动作"等于 BC anchor，改用自监督 action ranking。

## 相关概念
- [[SAC]]
- [[2-强化学习|强化学习]]
- [[TD-MPC]]
