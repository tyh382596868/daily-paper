---
type: concept
aliases: [门控循环单元, Gated Recurrent Unit]
---

# GRU（Gated Recurrent Unit）

## 定义

门控循环单元（GRU）是一种改进的循环神经网络单元，通过重置门和更新门控制信息流动，相比 LSTM 参数更少、计算更高效，适用于序列到序列的动力学建模任务。

## 数学形式

$$
\begin{aligned}
r_t &= \sigma(W_r [h_{t-1}, x_t]) \\
z_t &= \sigma(W_z [h_{t-1}, x_t]) \\
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

## 核心要点

1. **重置门 $r_t$**: 控制历史隐状态对候选隐状态的贡献程度
2. **更新门 $z_t$**: 控制新旧隐状态的混合比例，类似记忆/遗忘
3. **相比 LSTM**: 减少了输出门，参数量更少，在小规模任务上表现相当甚至更好
4. **用于潜动力学**: 在潜空间的自回归滚出中，GRU 凭借紧凑性适合嵌入式部署

## 代表工作

- [[SkyJEPA]]: 使用单层 GRU（隐藏维度 24）作为轻量级潜动力学预测器，整体模型仅 99K 参数

## 相关概念

- [[RNN]]: GRU 所属的循环神经网络大类
- [[TCN]]: 另一类序列建模方法，SkyJEPA 中用于编码器
- [[JEPA]]: GRU 在 SkyJEPA 中承担 JEPA 潜预测器的角色
