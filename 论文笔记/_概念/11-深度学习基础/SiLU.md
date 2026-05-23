---
type: concept
aliases: [SiLU, Swish, Sigmoid Linear Unit]
---

# SiLU

## 定义

Sigmoid Linear Unit，也称 Swish-1，一种平滑非单调的激活函数：$\mathrm{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid。

## 数学形式

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

导数:
$$
\mathrm{SiLU}'(x) = \sigma(x) + x \sigma(x)(1 - \sigma(x))
$$

## 核心要点

1. **平滑**: 与 [[ReLU]] 不同处处可导，反传更稳
2. **非单调**: 在 $x < 0$ 处有微小负值（$\min \approx -0.278$），允许少量负激活通过
3. **由 NAS 发现**: Google Brain 2017 用神经架构搜索找出
4. **典型使用**: [[Transformer]] FFN、扩散模型 UNet、[[TRM]] 头 MLP 等
5. **vs GELU**: 两者形状几乎一致，SiLU 计算更便宜（只需一次 sigmoid）

## 代表工作

- Ramachandran et al., 2017: Searching for Activation Functions
- [[TRM]]: 成对头隐藏层使用 SiLU
- 大多数现代扩散模型 UNet

## 相关概念

- [[ReLU]]
- [[Softplus]]
- [[Transformer]]
