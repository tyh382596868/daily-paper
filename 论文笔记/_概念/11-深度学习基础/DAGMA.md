---
type: concept
aliases: [DAGMA, Directed Acyclic Graph using Matrix-exponential]
---

# DAGMA

## 定义
基于矩阵指数的因果图发现方法，将有向无环图（DAG）学习转化为连续优化问题，通过矩阵指数约束替代原始的组合约束，效率远超传统 NOTEARS 等方法。

## 数学形式
$$\min_{W} \mathcal{L}(W; X) \quad \text{s.t.} \quad h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

其中 $W$ 为加权邻接矩阵，$h(W)=0$ 等价于图为 DAG 的约束。

## 核心要点
1. 用矩阵指数 trace 作为 DAG 约束，比 NOTEARS 的约束更平滑易优化
2. 支持线性和非线性（MLP）因果模型
3. 在高维变量空间的计算效率显著优于早期方法

## 代表工作
- [[DAGMA]]: Bello et al., 2022，原始论文
- [[CRWM]]: 将 DAGMA 用于机器人技能学习的因果奖励设计

## 相关概念
- [[因果推断]] — 上层概念
- [[DAG]] — 有向无环图结构
