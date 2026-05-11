---
type: concept
aliases: [CLS Token, [CLS] token, classification token]
---

# CLS Token

## 定义

CLS Token 是 [[Transformer]] 序列开头的一个**可学习向量**，其在末层的输出聚合了整个序列的信息，常被用作"句子级 / 图像级"的紧凑表示。最早出现在 BERT，后被 [[ViT]] 沿用。

## 数学形式

输入序列前置 $x_{\text{cls}}\in\mathbb{R}^d$：

$$
z_0 = [x_{\text{cls}};\; x_1;\; \dots;\; x_N] + E_{\text{pos}},\quad
\text{representation} = z_L^{[\text{cls}]}
$$

经 $L$ 层后取 CLS 位置的输出作为全局表示。

## 核心要点

1. **聚合机制**: 通过 [[自注意力]]，CLS token 隐式地对所有 patch / token 做加权求和
2. **可替代方案**: 平均池化（mean pooling）也常用，效果接近
3. **在自监督中**: [[DINO]] / I-JEPA / [[LeWM]] 都用 CLS token 作为帧表示
4. **效率优势**: 只用 CLS token 时下游 token 数大幅下降——[[LeWM]] 的规划速度提升 ~200× 即源于此

## 代表工作

- BERT (Devlin et al., 2019): 引入 [CLS] token
- [[ViT]]: 把 CLS token 用于图像分类
- [[LeWM]]: 用 CLS token 作为世界模型潜表示

## 相关概念

- [[ViT]]
- [[Transformer]]
- [[自注意力]]
