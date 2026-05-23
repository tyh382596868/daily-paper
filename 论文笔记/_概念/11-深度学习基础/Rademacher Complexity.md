---
type: concept
aliases: [Rademacher Complexity, 拉德马赫复杂度]
---

# Rademacher Complexity

## 定义

一种衡量函数类**表达能力**的复杂度度量，用以给出**泛化误差**的上界。给定函数类 $\mathcal{F}$ 和样本 $S = \{x_1, \dots, x_n\}$，其经验 Rademacher 复杂度衡量该函数类能多好地拟合**随机的 ±1 标签**。

## 数学形式

经验 Rademacher 复杂度:
$$
\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma \left[ \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i) \right]
$$

其中 $\sigma_i$ 是独立的 ±1 等概率随机变量（Rademacher 变量）。

泛化界 (与 $\sqrt{1/n}$ 同阶):
$$
\mathbb{E}[L(\hat f)] - \min_{f \in \mathcal{F}} \mathbb{E}[L(f)] \le 2 \hat{\mathfrak{R}}_S(\mathcal{F}) + O\!\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)
$$

## 核心要点

1. **核心思想**: 函数类越能拟合噪声标签，泛化能力越差
2. **比 VC 维更紧**: 对实值函数也适用，泛化界通常更紧
3. **理论工具**: 神经网络泛化分析（[[Spectral Norm]] 界、[[Margin]] 界）的基础
4. **训练分布偏移**: 当训练对分布与推理分布不匹配（如 [[Horizon-Matched Supervision]] 失败），泛化界变松，复杂度难以控制风险

## 代表工作

- Bartlett & Mendelson, 2002: 原始定义
- Koltchinskii, 2001: 局部 Rademacher 复杂度
- Neyshabur et al., 2018: 神经网络 Rademacher 复杂度分析

## 相关概念

- [[VC 维]]
- [[泛化误差]]
- [[Horizon-Matched Supervision]]
