---
type: concept
aliases: [Centered Kernel Alignment, CKA]
---

# CKA (Centered Kernel Alignment)

## 定义
一种度量两个神经网络表征之间相似度的指标。在不同模型 / 不同层之间比较时，对正交变换不敏感、不依赖维度对齐，是比直接 cosine 更稳健的表征相似度度量。

## 数学形式
$$
\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}
$$

其中 $K, L$ 为两组表征的 Gram 矩阵，HSIC 为 Hilbert-Schmidt Independence Criterion。

## 核心要点
1. **对正交变换不变**：不会因为权重的旋转/排列而改变
2. **不要求维度匹配**：可比较不同宽度的层
3. **常用于**：模型对比（finetune 前后）、层间相似度分析、representation shift 测量

## 在 VLA 中的应用
- [[ElegantVLA]]：用 CKA 测 vision-LLM 表征的语义稳定性，决定是否需要重新跑感知
- [[VLA-Trace]]：用 CKA 测 VLA 在 adaptation 前后的 representation shift

## 相关概念
- [[Linear Probing]]、[[Probing]]
