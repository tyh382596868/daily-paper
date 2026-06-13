---
type: concept
aliases: [RRT-Connect, Rapidly-exploring Random Tree Connect, 双向 RRT]
---

# RRT-Connect

## 定义

RRT-Connect 是一种高效的采样型运动规划算法，通过同时从起始状态和目标状态各生长一棵随机树（RRT），并在两树相遇时完成路径规划，相较于单向 RRT 显著提升规划效率。

## 核心要点

1. **双向扩展**: 同时从起点和终点生长两棵 RRT，利用目标偏置加速收敛
2. **Connect 操作**: 每次迭代尝试将一棵树直接延伸到另一棵树的最近节点，而非仅扩展一步
3. **概率完备性**: 在连续空间中，随样本数趋于无穷时，若路径存在则一定能找到
4. **GPU 加速**: 在机器人操作研究中常利用 GPU 并行化采样和碰撞检测，大幅提升规划速度

## 数学形式

RRT-Connect 的核心 Connect 操作：

$$
q_\text{new} = \text{Steer}(q_\text{nearest}, q_\text{rand})
$$

反复执行直至 $q_\text{new} = q_\text{rand}$ 或发生碰撞，实现树的快速延伸。

## 代表工作

- [[Mana]]: 使用 GPU 加速的 RRT-Connect 为关节型工具操作生成预抓取运动轨迹

## 相关概念

- [[MPC]]
- [[IsaacLab]]
- [[Sim-to-Real]]
