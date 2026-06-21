---
type: concept
aliases: [Transfusion, 文图混合目标模型]
---

# Transfusion

## 定义

Transfusion 是一种在**单个 Transformer** 上统一「文本自回归 + 图像扩散」的多模态生成范式：文本 token 走 next-token 自回归交叉熵目标，图像走连续扩散/[[Flow Matching]] 目标，二者在同一条序列、同一套 backbone 上联合训练。与 [[Chameleon]] 把图像也离散化做纯自回归不同，Transfusion 保留图像的连续表征用扩散建模。

## 数学形式

$$
\mathcal{L} = \mathcal{L}_{\text{AR}}^{\text{text}} + \lambda\, \mathcal{L}_{\text{diffusion}}^{\text{image}}
$$

文本部分是自回归交叉熵，图像部分是扩散/流匹配去噪损失，混合权重 $\lambda$ 平衡两目标。

## 核心要点

1. 一套 Transformer backbone 同时承载离散（文本）与连续（图像）两种生成目标。
2. 图像用扩散而非离散 token，避免 VQ 量化损失，图像质量更高。
3. 是 [[Mixture-of-Transformers]] 的两大实验设定之一（另一个是 [[Chameleon]]），用于验证模态解耦对「不同模态不同目标」范式同样有效。

## 代表工作

- [[Mixture-of-Transformers]]：在 Transfusion 设定下，MoT 7B 用约 1/3 FLOPs 匹配 dense 图像质量。

## 相关概念

- [[Chameleon]]
- [[Diffusion Model]]
- [[Flow Matching]]
- [[Mixture-of-Transformers]]
- [[Unified Multimodal Model]]
