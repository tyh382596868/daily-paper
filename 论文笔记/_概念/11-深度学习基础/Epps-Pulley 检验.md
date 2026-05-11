---
type: concept
aliases: [Epps-Pulley test, Epps–Pulley test, Epps Pulley]
---

# Epps-Pulley 检验

## 定义

Epps & Pulley (1983) 提出的一维**正态性检验**，通过比较经验[[特征函数]]与标准正态特征函数的加权 $L^2$ 距离来判断样本是否来自 $\mathcal{N}(0, 1)$。具有闭式解析公式，因此**端到端可微**，非常适合作为深度学习中的分布匹配损失。

## 数学形式

$$
T = \int_{-\infty}^{\infty} w(t)\, |\phi_N(t) - \phi_0(t)|^2 \, dt
$$

其中：
- $\phi_N(t) = \frac{1}{N} \sum_{n=1}^N e^{i t h_n}$ 是基于 $N$ 个样本 $\{h_n\}$ 的经验特征函数
- $\phi_0(t) = e^{-t^2/2}$ 是标准正态的特征函数
- $w(t)$ 通常取高斯权重，使积分有闭式解

## 核心要点

1. **可微 + 闭式**：相比 KS、Anderson–Darling 等检验，Epps–Pulley 的统计量可以闭式表达，方便反向传播
2. **对偏度/峰度敏感**：通过特征函数捕捉所有阶矩信息
3. **配合 Cramér–Wold 定理**：用于多元高斯检验时只需对每个 1D 投影跑一次

## 代表工作

- Epps & Pulley, 1983: 原始论文
- [[SIGReg]] / [[LeWM]]: 在自监督表示学习中作为可微正则项

## 相关概念

- [[特征函数]]
- [[Cramér-Wold 定理]]
- [[SIGReg]]
