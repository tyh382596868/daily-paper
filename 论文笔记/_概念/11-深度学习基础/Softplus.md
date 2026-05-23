---
type: concept
aliases: [Softplus, 平滑 ReLU]
---

# Softplus

## 定义

ReLU 的平滑近似，输出始终为正且处处可导：$\mathrm{Softplus}(x) = \log(1 + e^x)$。常用于需要**严格正输出**的网络头（如距离、方差预测）。

## 数学形式

$$
\mathrm{Softplus}(x) = \log(1 + e^x)
$$

导数即 sigmoid:
$$
\mathrm{Softplus}'(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

带温度变体:
$$
\mathrm{Softplus}_\beta(x) = \frac{1}{\beta} \log(1 + e^{\beta x})
$$

当 $\beta \to \infty$ 退化为 [[ReLU]]。

## 核心要点

1. **严格正输出**: 适合预测距离、方差、概率比例等天然非负的量
2. **数值稳定实现**: $\log(1 + e^x)$ 在大 $x$ 处会溢出，PyTorch 用 $\max(x, 0) + \log(1 + e^{-|x|})$
3. **梯度光滑**: 在 $x=0$ 附近梯度连续过渡，比 ReLU 的硬拐点训练更稳
4. **典型应用**: [[TRM]] 成对头输出层（保证距离 ≥0）、高斯混合密度网络 (MDN) 的方差预测、[[VAE]] 的方差输出

## 代表工作

- [[TRM]]: 距离头输出层
- Dugas et al., 2001: Softplus 在 MDN 中的应用

## 相关概念

- [[ReLU]]
- [[SiLU]]
- [[Pairwise Ranking Head]]
