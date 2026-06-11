---
type: concept
aliases: [robomimic, RoboMimic Benchmark]
---

# RoboMimic

## 定义
斯坦福大学提出的大规模机器人操作模仿学习 benchmark，包含多种操作任务（Lift、Can、Square 等）和不同质量水平的专家演示数据，用于评估模仿学习和离线强化学习算法。

## 数学形式

$$
\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}[R(\tau)]
$$

通过模仿专家演示 $\mathcal{D} = \{(s_t, a_t)\}$ 学习策略 $\pi$。

## 核心要点
1. 提供多质量层级的演示数据（熟练/一般操作员），方便研究数据质量对策略的影响
2. 包含 Lift、Can、Square、Transport、ToolHang 等 5 种操作任务
3. 广泛用于 Diffusion Policy、ACT、π0 等策略的标准化评估

## 代表工作
- [[RoboScape]]: 在 RoboMimic Lift 任务上验证合成数据增益（200 轨迹真实数据 + 合成数据），用于策略评估相关性实验（Pearson 0.953）

## 相关概念
- [[Diffusion Policy]]
- [[Action Chunking]]
- [[LIBERO]]
