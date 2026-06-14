---
type: concept
aliases: [Fourier Feature Encoding, 傅里叶特征编码, 随机傅里叶特征]
---

# Fourier 特征编码

## 定义
将低维输入映射到高维周期性特征空间的编码方式，使神经网络更容易学习高频函数，常用于状态型传感器信号的预处理。

## 数学形式

$$
\gamma(x) = \left[\sin(2\pi \mathbf{B} x),\, \cos(2\pi \mathbf{B} x)\right]
$$

其中 $\mathbf{B}$ 为随机采样或可学习的频率矩阵。

## 核心要点
1. 解决神经网络对高频信号学习困难的问题（"spectral bias"）
2. 在 [[FTP-1]] 中用于对力矩等状态型传感器信号进行预处理，再经 MLP 编码至 [[MTTS]] 令牌
3. 也广泛用于 NeRF 中的位置编码

## 代表工作
- [[FTP-1]]: 对状态型触觉传感器（力矩）使用 Fourier 编码 + MLP 进行令牌化

## 相关概念
- [[MTTS]]
- [[MLP]]
- [[触觉传感器]]
