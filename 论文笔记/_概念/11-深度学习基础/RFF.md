---
type: concept
aliases: [Random Fourier Features, 随机傅里叶特征, Fourier Position Encoding]
---

# RFF（随机傅里叶特征）

## 定义
RFF（Random Fourier Features）是一种将低维输入映射到高维傅里叶特征空间的技术，让模型能够学习高频函数，常用于位置编码和核函数近似。

## 数学形式
对输入 $\mathbf{x} \in \mathbb{R}^d$，RFF 映射为：
$$\gamma(\mathbf{x}) = \left[\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})\right]$$

其中 $\mathbf{B} \in \mathbb{R}^{m \times d}$ 从高斯分布 $\mathcal{N}(0, \sigma^2)$ 随机采样，$\sigma$ 控制频率范围。

等价核函数近似：
$$k(\mathbf{x}, \mathbf{y}) \approx \gamma(\mathbf{x})^\top \gamma(\mathbf{y})$$

## 核心要点
1. NeRF 中用于编码 3D 坐标，使网络能学习高频几何细节
2. 在点云操作策略中（如 FourierManip），用于编码点坐标，提升精度
3. 频率参数 $\sigma$ 决定能学习的最高频率，需要根据任务调整
4. 计算简单，维度可控，是 SPE（Sinusoidal Position Encoding）的随机版本

## 代表工作
- [[NeRF]]: 将 RFF 用于 3D 坐标编码，开创高频位置编码在 3D 中的应用
- [[FourierManip]]: 用 RFF 增强点云操作策略的精度

## 相关概念
- [[NeRF]]
- [[3D mRoPE]]
- [[AdaLN]]
