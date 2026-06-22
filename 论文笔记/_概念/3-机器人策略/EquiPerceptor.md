---
type: concept
aliases: [等变感知器, Token-level Frame Averaging]
---

# EquiPerceptor

## 定义

EquiVLA 的视觉模块，将 [[Frame Averaging]] 从全局池化扩展到 ViT patch token 序列，在不修改冻结 VLM 权重的情况下产生近似 SO(2) 等变的视觉 token 表示。

## 数学形式

**等变流**：

$$
\mathbf{z}^{\mathrm{eq}}(\mathbf{x}) = \frac{1}{|G|} \sum_{h \in G} \bigl[\tau(h^{-1}) \otimes \rho_{\mathrm{reg}}(h^{-1})\bigr] \cdot f_\theta(h \cdot \mathbf{x})
$$

**不变流**：

$$
\mathbf{z}^{\mathrm{inv}}(\mathbf{x}) = \frac{1}{|G|} \sum_{h \in G} f_\theta(h \cdot \mathbf{x})
$$

## 核心要点

1. **双流设计**：等变流（保留空间位置关系）+ 不变流（送入冻结 VLM 作为全局上下文）
2. **空间置换** $\tau$：将旋转后的 patch 中心重映射回规范网格位置，解决 token 位置错位问题
3. **等变适配器**：用不变门控 $\alpha^{\mathrm{reg}}$ 融合等变流与 VLM context token，保证输出等变性
4. **近似误差**：C4 在正方形网格上精确等变（$\Delta=0$），C8 有有界误差

## 代表工作

- [[EquiVLA]]: EquiPerceptor 的提出论文

## 相关概念

- [[Frame Averaging]]
- [[SO(2)等变性]]
- [[EquiActor]]
- [[ViT]]
