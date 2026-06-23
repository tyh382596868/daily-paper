---
type: concept
aliases: [时域卷积网络, Temporal Convolutional Network, 时间卷积网络]
---

# TCN（Temporal Convolutional Network）

## 定义

时域卷积网络（TCN）是一类专为序列建模设计的卷积神经网络，通过因果卷积和扩张卷积实现长序列的并行处理，同时保证时间因果性（当前输出仅依赖过去输入）。

## 数学形式

$$
y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}
$$

其中 $d$ 为扩张率（dilation rate），$K$ 为卷积核大小，感受野为 $O(d \cdot K)$。

## 核心要点

1. **因果卷积（Causal Convolution）**: 卷积只能看到当前及过去时刻，保证时间因果性
2. **扩张卷积（Dilated Convolution）**: 指数级增大感受野，以 $O(1)$ 深度覆盖长历史
3. **并行训练**: 不同于 RNN 的序列依赖，TCN 可在时间维度全并行训练
4. **残差连接**: 深层 TCN 常配合残差块缓解梯度消失

## 代表工作

- [[SkyJEPA]]: 用 TCN 编码状态历史和动作历史为上下文潜变量

## 相关概念

- [[RNN]]: 循环神经网络，另一类序列建模方法，计算串行
- [[GRU]]: TCN 的序列建模竞争方案，SkyJEPA 中用于潜动力学预测器
- [[Transformer]]: 基于注意力的序列建模，对长序列注意力计算复杂度高
