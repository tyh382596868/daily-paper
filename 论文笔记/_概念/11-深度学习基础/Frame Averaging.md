---
type: concept
aliases: [FA, 帧平均, 群平均]
---

# Frame Averaging

## 定义

Frame Averaging（FA）是一种将任意函数对称化为群等变函数的技术：对群 $G$ 的所有元素，分别变换输入、计算输出、再逆变换后平均，从而在不修改原函数权重的情况下获得等变性。

## 数学形式

**标准 FA（全局池化版本）**：

$$
f^{\mathrm{FA}}(x) = \frac{1}{|G|} \sum_{h \in G} \rho_{\mathrm{out}}(h^{-1}) \cdot f(\rho_{\mathrm{in}}(h) \cdot x)
$$

**Token 级 FA（EquiPerceptor 扩展版）**：

$$
\mathbf{z}^{\mathrm{eq}}(\mathbf{x}) = \frac{1}{|G|} \sum_{h \in G} \bigl[\tau(h^{-1}) \otimes \rho_{\mathrm{reg}}(h^{-1})\bigr] \cdot f_\theta(h \cdot \mathbf{x})
$$

## 核心要点

1. **免权重修改**：原模型权重完全冻结，等变性通过输入/输出变换的平均获得
2. **计算代价**：需要 $|G|$ 次前向推理（如 $C_8$ 需 8 次），是主要延迟来源
3. **空间 token 挑战**：ViT patch token 带有位置索引，旋转后需要额外空间置换 $\tau$ 对齐，否则直接平均会破坏空间结构
4. **近似等变**：当空间置换 $\tau$ 只能近似时（如非 90° 倍数旋转），误差有界于 $\Delta \cdot B(x)$（Theorem 3）

## 代表工作

- [[EquiVLA]]: 将 FA 从全局池化扩展到 ViT patch token 序列，提出 Token 级 Frame Averaging

## 相关概念

- [[SO(2)等变性]]
- [[EquiPerceptor]]
- [[ViT]]
