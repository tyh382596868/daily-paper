---
type: concept
aliases: [MCTS, Monte Carlo Tree Search, 蒙特卡洛树搜索]
---

# MCTS（蒙特卡洛树搜索）

## 定义

蒙特卡洛树搜索（Monte Carlo Tree Search）是一种通过随机采样构建搜索树来找到最优决策的启发式算法，广泛应用于博弈和规划任务。

## 数学形式

UCB（Upper Confidence Bound）选择准则：

$$
a^* = \mathop{\mathrm{arg\,max}}_{a} \left[ Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]
$$

其中 $Q(s,a)$ 为动作价值估计，$N(s)$ 为状态访问次数，$N(s,a)$ 为动作访问次数，$C$ 为探索系数。

## 核心要点

1. **四阶段**: 选择（Selection）→ 扩展（Expansion）→ 模拟（Simulation）→ 回传（Backpropagation）。
2. **无需完整模型**: 仅需前向采样即可运行。
3. 在 AlphaGo/AlphaZero 中与神经网络结合取得突破性成果。

## 代表工作

- [[ROAD-VLA]]: 消融实验中尝试用 MCTS 提示作为文本特权信息，结果证明对 VLA 适应无效（~75.8% vs 91.5%）

## 相关概念

- [[优势函数（Advantage Function）]]
- [[强化学习]]
