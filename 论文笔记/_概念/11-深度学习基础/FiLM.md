---
type: concept
aliases: [Feature-wise Linear Modulation, FiLM, 特征线性调制]
---

# FiLM（Feature-wise Linear Modulation）

## 定义

Feature-wise Linear Modulation（FiLM）是一种通用的条件注入机制：对中间特征图按通道做仿射变换（缩放 + 偏移），缩放和偏移系数由条件信号（如类别标签、文本嵌入、扩散时间步等）动态生成，从而将外部条件信息注入神经网络中间层。

## 数学形式

$$
\mathrm{FiLM}(h \mid c) = \gamma(c) \odot h + \beta(c)
$$

其中 $h$ 为中间特征，$c$ 为条件信号，$\gamma(c), \beta(c)$ 为由小型网络从 $c$ 预测的逐通道缩放和偏移系数，$\odot$ 表示逐元素乘法。

## 核心要点

1. **轻量灵活**: 条件网络通常是简单 MLP，额外参数极少
2. **通用性**: 可注入任意标量或向量条件（时间步、类别、语言嵌入）
3. **扩散模型中的应用**: 在扩散/Flow Matching 模型中广泛用于注入去噪时间步 $t$，替代早期的 positional embedding 相加方式
4. **与 AdaLN 的关系**: [[AdaLN]]（Adaptive Layer Norm）是 FiLM 的归一化后特例，先做 LayerNorm 再用 FiLM 缩放偏移

## 代表工作

- Perez et al. (2018): 原始 FiLM 论文，用于视觉推理
- [[Diffusion Policy]]: 用 FiLM 注入扩散时间步
- [[APT]]: 行动专家中使用 FiLM 注入扩散时间步 $t$

## 相关概念

- [[AdaLN]]
- [[Diffusion Policy]]
- [[Action Expert]]
