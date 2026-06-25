---
type: concept
aliases: [Advantage Function, 优势函数, 优势估计]
---

# 优势函数（Advantage Function）

## 定义

优势函数 $A^\pi(s, a)$ 衡量在状态 $s$ 下执行动作 $a$ 相比于策略 $\pi$ 的平均水平好多少。

## 数学形式

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

其中 $Q^\pi(s,a) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0=s, a_0=a]$ 为 Q 值函数，$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$ 为状态值函数。

## 核心要点

1. $A > 0$：该动作优于平均策略，应增加其概率。
2. $A < 0$：该动作劣于平均策略，应降低其概率。
3. $A = 0$：该动作与平均水平持平。
4. 优势函数消除了 Q 值中与动作无关的基线项，减小方差。

## 代表工作

- [[PPO]]: 使用裁剪的优势估计做策略更新
- [[GAE（广义优势估计）]]: 通过 λ-加权 TD 误差估计优势
- [[ROAD-VLA]]: 用校准混合优势权重构造 proximal teacher 分布

## 相关概念

- [[PPO]]
- [[GAE（广义优势估计）]]
- [[强化学习]]
