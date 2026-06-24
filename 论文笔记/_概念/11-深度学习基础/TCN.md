---
type: concept
aliases: [时序卷积网络, Temporal Convolutional Network, 因果卷积]
---

# TCN

## 定义

TCN（Temporal Convolutional Network，时序卷积网络）是一种用于序列建模的 1D 卷积架构，通过因果膨胀卷积（causal dilated convolution）在不引入未来信息的前提下高效捕获长程时序依赖。

## 数学形式

膨胀因果卷积（dilated causal convolution，膨胀率 $d$，核大小 $k$）：

$$
(f * g)(t) = \sum_{i=0}^{k-1} f(i) \cdot g(t - d \cdot i)
$$

感受野大小：$\mathrm{RF} = 1 + (k-1) \cdot d$；通过多层叠加指数增大感受野。

## 核心要点

1. 因果性：只依赖当前及过去时刻，适合在线序列预测
2. 膨胀卷积以 $O(\log T)$ 层数实现 $O(T)$ 感受野，比 RNN 并行效率更高
3. 无递归结构，可在训练时完全并行计算梯度
4. 相比 [[GRU]] / [[RNN]]，训练更稳定，但对超长序列的记忆能力依赖网络深度

## 代表工作

- [[SkyJEPA]]: 用 TCN 作为状态编码器（通道 $[8,8,16]$）和动作编码器（通道 $[4,4,8]$），将历史序列压缩为潜在向量

## 相关概念

- [[GRU]]
- [[JEPA]]
