---
type: concept
aliases: [DCT, Discrete Cosine Transform, 离散余弦变换]
---

# 离散余弦变换 (DCT)

## 定义

将有限长度实数序列从时域变换到频域的正交变换，输出余弦基函数的线性组合系数；在信号处理（JPEG、MP3）和机器学习（动作 Token 化）中广泛应用。

## 数学形式

对于长度为 $N$ 的序列 $x[n]$，DCT-II（最常用变体）定义为：

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cos\!\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right], \quad k = 0, 1, \ldots, N-1
$$

## 核心要点

1. **能量集中**: 大多数信号的能量集中在低频系数（$k$ 小的分量），高频系数接近零，适合截断压缩。
2. **可逆变换**: 存在 IDCT（逆 DCT），可精确还原原始序列。
3. **运动信号友好**: 机器人关节轨迹通常平滑（低频主导），DCT 可用极少系数近似重建。
4. **与 FFT 对比**: DCT 只涉及余弦基（实数），无需处理虚部，更适合实值序列。

## 代表工作

- [[DCT 动作 Token 化]]: 将 DCT 应用于机器人动作 chunk 量化
- [[UniVLA-ICLR2026]]: 使用 DCT+BPE 将动作统一为离散 Token

## 相关概念

- [[DCT 动作 Token 化]]
- [[字节对编码 (BPE)]]
- [[Action Chunking]]
