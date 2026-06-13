---
type: concept
aliases: [Voyager world model]
---

# Voyager

## 定义

一种基于 RGB 点云空间记忆的视频世界模型，通过维护彩色 3D 点云实现跨 chunk 的场景一致性，是 [[Latent Spatial Memory]] 方法的主要对比基线之一。

## 核心要点

1. **RGB 点云记忆**: 以 $(\mathbf{p}_i, \mathbf{c}_i)$ 形式存储 3D 彩色点，每步条件化需 rasterise → VAE 编码的往返
2. **WorldScore 性能**: Avg 66.08，3D 一致性 81.56
3. **RealEstate10K 性能**: PSNR 17.79，SSIM 0.636，LPIPS 0.297

## 代表工作

- [[Mirage]]: 提出 Latent Spatial Memory，效率和质量均优于 Voyager

## 相关概念

- [[Latent Spatial Memory]]
- [[视频扩散模型]]
- [[RealEstate10K]]
