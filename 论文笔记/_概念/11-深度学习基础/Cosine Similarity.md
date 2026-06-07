---
type: concept
aliases: [余弦相似度, 余弦距离, Cosine Distance]
---

# Cosine Similarity

## 定义
衡量两个向量方向相似程度的相似度度量，取值范围 $[-1, 1]$，与向量模长无关。常用于特征对齐、蒸馏、检索等场景。

## 数学形式

$$
\mathcal{S}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2}
$$

对应的 cosine 距离损失：
$$
\mathcal{L}_{cos} = 1 - \mathcal{S}(\mathbf{a}, \mathbf{b})
$$

Patch-level 平均（如 [[3DThinkVLA]] 几何对齐）：
$$
\mathcal{S}(\mathcal{F}^A, \mathcal{F}^B) = \frac{1}{N}\sum_{i=1}^{N} \mathcal{S}(\mathcal{F}^A_i, \mathcal{F}^B_i)
$$

## 核心要点
1. **方向敏感、模长不敏感**: 适合表征空间对齐，因为表征的"语义方向"比"激活强度"更稳定
2. **天然归一化**: 不像 L2 / MSE 会被 outlier 主导
3. **梯度温和**: 接近 1 时梯度小，对稳定收敛友好
4. **在蒸馏中常胜过 KL/MSE**: 特别是 hidden-state 蒸馏，对齐方向即可，无需强制激活相等

## 代表工作
- [[CLIP]]: 用 cosine 做图文对齐
- [[DINO]] / [[DINOv2]]: cosine 自蒸馏
- [[3DThinkVLA]]: cosine 用于几何对齐 + reasoning 蒸馏

## 相关概念
- [[L2 损失]]
- [[KL 散度]]
- [[Latent Alignment]]
