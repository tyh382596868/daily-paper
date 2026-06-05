---
type: concept
aliases: [VQ-VAE, Vector Quantized VAE, 向量量化变分自编码器]
---

# VQ-VAE（Vector Quantized Variational Autoencoder）

## 定义

一种用离散 codebook 量化连续 latent 的变分自编码器，将 encoder 输出的连续向量映射到最近邻的离散 code，用于生成模型、token 化图像/音频等。

## 数学形式

给定 encoder 输出 $z_e(x)$，离散化为：
$$
z_q(x) = e_k,\quad k = \arg\min_j \| z_e(x) - e_j \|_2
$$
其中 $\{e_j\}$ 是 codebook。

## 核心要点

1. **离散表征**: 把连续 latent 压缩为离散 token，便于自回归建模。
2. **量化误差**: 离散化引入信息损失，存在 codebook collapse 风险。
3. **直通梯度**: 用 straight-through estimator 让梯度跨过量化操作。
4. **应用**: VQGAN、自回归图像生成、音频 token、AffordanceVLA 中显式避开它选择连续 latent。

## 代表工作

- VQ-VAE / VQ-VAE-2 原文
- VQGAN
- [[AffordanceVLA]]: 显式选择 [[FLUX]] 连续 VAE 避免量化误差

## 相关概念

- [[VAE]]
- [[3D Causal VAE]]
- [[FLUX]]
- [[Codebook Collapse]]
