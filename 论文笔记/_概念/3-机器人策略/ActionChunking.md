---
type: concept
aliases: [Action Chunking, 动作块, 动作预测块, Chunk-based Action Prediction]
---

# Action Chunking

## 定义
Action Chunking 是机器人策略中的一种输出策略，模型在每个决策步预测未来 $H$ 步的连续动作序列（一个"块"），而非单步动作，以减少推理频率并提升时序一致性。

## 数学形式

给定当前观测 $\mathbf{o}_t$ 和状态 $\mathbf{q}_t$，预测动作块：

$$
\mathbf{a}_{t:t+H} = \pi_\theta(\mathbf{o}_t, \mathbf{q}_t, \mathbf{l}_t)
$$

执行时每步取对应时间步的动作 $\mathbf{a}_t$，在 $H$ 步后重新推理更新块。

## 核心要点

1. **减少推理频率**: 每隔 $H$ 步调用一次策略网络，降低实时推理开销
2. **时序平滑**: 预测整段轨迹而非单步，自然引入时序约束，减少抖动
3. **与流匹配结合**: Flow Matching 训练时直接对整个动作块施加噪声和去噪，联合建模时序关联
4. **Horizon 权衡**: $H$ 过大影响实时响应（ALLEX 用 40 步，FR3 用 16 步）

## 代表工作

- [[ACT]]: 首先在模仿学习中广泛采用 Action Chunking
- [[pi0]]: 将 Action Chunking 与 Flow Matching 结合用于 VLA
- [[RLDX-1]]: ALLEX 用 $H=40$，FR3 用 $H=16$

## 相关概念

- [[Flow Matching]]
- [[VLA]]
- [[MSAT]]
