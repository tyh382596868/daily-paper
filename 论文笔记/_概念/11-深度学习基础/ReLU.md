---
type: concept
aliases: [Rectified Linear Unit, 修正线性单元]
---

# ReLU

## 定义

最常见的非线性激活函数：负值置零，正值保持不变。

## 数学形式

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

## 核心要点

1. 计算极简，反向传播只需 0/1 指示函数
2. 在零点不可导但实践中无影响（次梯度）
3. "死亡 ReLU" 问题：大梯度后神经元永远输出 0，可用 LeakyReLU/GELU/SiLU 缓解
4. 在 [[线性注意力]] 变体中常作为 Key/Query 的特征映射，保证非负性

## 代表工作

- [[SANA-WM]]: Key 路径 `ReLU(RMSNorm(K))` 以确保 [[Gated DeltaNet]] 转移矩阵谱范数非负

## 相关概念

- [[Transformer]]
- [[MLP]]
