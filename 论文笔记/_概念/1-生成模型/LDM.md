---
type: concept
aliases: [Latent Diffusion Model, 潜扩散模型]
---

# LDM

## 定义
在 VAE 编码的潜空间（而非像素空间）中运行扩散过程的生成模型，大幅降低计算成本同时保持生成质量。

## 数学形式
$$z_0 = \mathcal{E}(x), \quad z_T \sim \mathcal{N}(0,I)$$
$$p_\theta(z_{t-1}|z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \sigma_t^2 I)$$
解码：$\hat{x} = \mathcal{D}(z_0)$

## 核心要点
1. VAE encoder/decoder 先行训练，冻结后训练扩散 U-Net/DiT
2. 潜空间维度远低于像素空间，训练推理均加速 4-16×
3. Stable Diffusion 系列均基于 LDM

## 代表工作
- Rombach et al., 2022: LDM 原始论文（CVPR）

## 相关概念
- [[DiT]]
- [[DINO]]
- [[SigLIP]]
