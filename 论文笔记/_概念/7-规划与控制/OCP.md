---
type: concept
aliases: [Optimal Control Problem, 最优控制]
---

# OCP

## 定义
最优控制问题（Optimal Control Problem）：在系统动力学约束下，寻找使代价函数最小的控制输入序列，是 MPC 和轨迹优化的理论基础。

## 数学形式
$$\min_{u(\cdot)} \int_0^T \ell(x(t), u(t)) dt + \Phi(x(T))$$
$$\text{s.t.} \quad \dot{x} = f(x, u), \quad x(0) = x_0, \quad g(x, u) \leq 0$$

## 核心要点
1. 三要素：状态变量 $x$、控制输入 $u$、代价函数 $\ell$
2. 常用求解方法：[[MPC]]（滚动时域）、直接配置法、间接法（Pontryagin 最大值原理）
3. 对人形机器人全身控制（whole-body control）：OCP 保证动力学一致性，生成高质量 demonstration
4. 与 RL 的对比：OCP 需要精确动力学模型，RL 从交互中学习

## 代表工作
- [[WOLF-VLA]]: 用 OCP 生成动力学一致的人形运动 demonstration

## 相关概念
- [[MPC]]
- [[admittance control]]
- [[MPPI]]
