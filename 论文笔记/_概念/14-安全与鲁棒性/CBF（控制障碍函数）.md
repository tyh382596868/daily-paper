---
type: concept
aliases: [Control Barrier Function, CBF, 控制障碍函数]
---

# CBF（控制障碍函数）

## 定义

Control Barrier Function（CBF，控制障碍函数）是一种基于 Lyapunov 稳定性理论的安全约束方法，通过构造超平面函数 $h(x)$ 保证系统状态始终处于安全集合内（$h(x) \geq 0$），常用于实时安全过滤控制指令。

## 数学形式

对于安全集合 $\mathcal{S} = \{x \mid h(x) \geq 0\}$，CBF 约束要求：

$$
\dot{h}(x, u) + \alpha(h(x)) \geq 0
$$

其中 $\alpha$ 为 class-$\mathcal{K}$ 函数。实际控制中常用 QP（二次规划）形式：

$$
u^* = \arg\min_{u} \|u - u_{\text{ref}}\|^2 \quad \text{s.t.} \quad h(x, u) \geq 0
$$

## 核心要点

1. **安全集合前向不变性**：CBF 保证系统一旦在安全集内，后续状态永远不会离开安全集
2. **闭形式投影**：对静态约束（如 CoP 裕量），可直接做闭形式投影，无需迭代 QP 求解
3. **与 RL 结合**：可用于对 RL 策略输出进行实时安全过滤，或对训练数据进行安全预处理

## 代表工作

- [[HANDOFF]]: 用静态 CoP 裕量 CBF 投影过滤重定向动捕数据，确保 WBC Teacher 训练数据动力学可行

## 相关概念

- [[PPO]]
- [[全身控制]]
- [[14-安全与鲁棒性]]
