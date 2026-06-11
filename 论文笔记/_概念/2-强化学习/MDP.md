---
type: concept
aliases: [Markov Decision Process, 马尔科夫决策过程, 马尔可夫决策过程]
---

# MDP

## 定义

马尔科夫决策过程（Markov Decision Process）是强化学习和序贯决策的数学框架，用五元组 $(S, A, P, R, \gamma)$ 描述智能体与环境的交互。

## 数学形式

MDP 定义为 $(S, A, P, R, \gamma)$：

$$
P(s_{t+1} \mid s_t, a_t) \text{（转移概率）}, \quad r_t = R(s_t, a_t) \text{（即时奖励）}
$$

目标：最大化累积折扣回报 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$

## 核心要点

1. **马尔科夫性**: 下一状态只依赖当前状态和动作，与历史无关
2. **观测 vs 状态**: 机器人任务中通常处理部分可观测 MDP（POMDP），用历史观测近似状态
3. **交织序列**: UniVLA 等 VLA 模型将 MDP 轨迹建模为 $(o_1, a_1, o_2, a_2, \ldots)$ 交织序列
4. **策略**: $\pi(a \mid s)$ 是状态到动作的映射，VLA 通过自回归生成实现

## 代表工作

- [[UniVLA]]: 在 MDP 框架下将观测帧和动作 token 交织排列进行自回归建模
- [[Diffusion Policy]]: 基于 MDP 的扩散策略学习

## 相关概念

- [[VLA]]: 机器人操作中求解 MDP 的主流框架
- [[Action Chunking]]: MDP 中的多步动作预测策略
- [[世界模型]]: 学习 MDP 转移函数 $P(s_{t+1} \mid s_t, a_t)$
