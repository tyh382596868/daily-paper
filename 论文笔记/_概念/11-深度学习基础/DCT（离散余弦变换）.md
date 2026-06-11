---
type: concept
aliases: [DCT, Discrete Cosine Transform, 离散余弦变换]
---

# DCT（离散余弦变换）

## 定义

一种将离散信号从时域/空间域变换到频域的正交变换，仅使用余弦函数作为基函数，广泛用于信号压缩（JPEG、MP3）和机器学习中的特征提取。

## 数学形式

对长度为 $N$ 的信号 $x_n$，DCT-II（最常用）：

$$
X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right], \quad k = 0, 1, \ldots, N-1
$$

逆变换（IDCT）：

$$
x_n = \frac{1}{N} X_0 + \frac{2}{N} \sum_{k=1}^{N-1} X_k \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]
$$

## 核心要点

1. **能量集中**: 大部分信号能量集中在低频系数，高频系数接近零 → 易于压缩
2. **无相位分量**: 纯实数变换，计算简单
3. **时序压缩**: 对动作轨迹做 DCT，低频系数表示平滑运动趋势，比原始时序更紧凑
4. **可逆性**: 信息无损（不考虑量化），保留全部动作信息

## 代表工作

- [[FAST Action Tokenizer]]: 将 DCT 用于机器人动作序列压缩分词
- [[UniVLA]]: 使用 FAST（DCT-based）动作分词器的 VLA 系统

## 相关概念

- [[FAST Action Tokenizer]]: DCT 的主要应用场景
- [[Action Chunking]]: DCT 处理的序列粒度
- [[Discrete Tokenization]]: DCT + 量化后得到离散 token
