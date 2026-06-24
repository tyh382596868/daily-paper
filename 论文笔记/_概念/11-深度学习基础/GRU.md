---
type: concept
aliases: [门控循环单元, Gated Recurrent Unit]
---

# GRU

## 定义

GRU（Gated Recurrent Unit，门控循环单元）是一种简化的循环神经网络单元，通过更新门（update gate）和重置门（reset gate）控制历史信息的保留与遗忘，在参数量少于 LSTM 的情况下实现类似的长程依赖建模能力。

## 数学形式

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1})
$$

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1})
$$

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}))
$$

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

其中 $\mathbf{z}_t$ 为更新门，$\mathbf{r}_t$ 为重置门，$\odot$ 为逐元素乘法。

## 核心要点

1. 比 LSTM 少一个门（无输出门），参数约减少 25%，计算效率更高
2. 更新门 $\mathbf{z}_t$ 同时控制遗忘和输入，避免梯度消失
3. 适合中等长度序列；极长序列可能仍不如 Transformer
4. 在轻量级嵌入式部署场景中常作为序列预测骨干

## 代表工作

- [[SkyJEPA]]: 用单层 GRU（隐藏维度 24）作为潜在动力学预测器，递归展开 20 步（1.0 s at 20 Hz）

## 相关概念

- [[TCN]]
- [[RNN]]
- [[JEPA]]
