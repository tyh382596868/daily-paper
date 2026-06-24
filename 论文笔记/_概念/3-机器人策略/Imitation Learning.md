---
type: concept
aliases: [模仿学习, IL, Behavior Cloning, BC, 行为克隆]
---

# Imitation Learning

## 定义

模仿学习（Imitation Learning）是通过观察专家示范数据学习策略的机器学习范式，无需显式奖励函数，直接将专家行为映射到策略输出，是机器人操作和 VLA 训练的主流方法。

## 数学形式

行为克隆（最简单形式，最大似然）：

$$
\mathcal{L}_{BC} = -\mathbb{E}_{(s,a) \sim \mathcal{D}} [\log \pi_\theta(a \mid s)]
$$

## 核心要点

1. **行为克隆（BC）**：直接回归专家动作，简单高效但存在分布偏移问题
2. **DAgger**：交互式数据收集缓解分布偏移
3. **逆强化学习（IRL）**：从示范中恢复奖励函数，再用 RL 优化
4. **VLA 中的应用**：主流 VLA（π₀、GR00T、OpenVLA）均采用 BC 形式的模仿学习目标

## 代表工作

- [[π₀]]: 以 flow-matching 形式实现 VLA 的模仿学习
- [[G3VLA]]: 在模仿学习目标中加入几何蒸馏损失作为辅助正则化

## 相关概念

- [[VLA|Vision-Language-Action]]
- [[Diffusion Policy]]
- [[Action Chunking]]
