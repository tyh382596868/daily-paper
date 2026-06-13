---
type: concept
aliases: [Video Memory]
---

# VMem

## 定义

一种视频记忆（Video Memory）方案，用于视频世界模型的跨 chunk 一致性，是 [[Mirage]] 的对比基线之一。

## 核心要点

1. **RealEstate10K 性能**: PSNR 14.62，SSIM 0.522，LPIPS 0.426
2. **与 Mirage 对比**: 在所有指标上均弱于 Mirage，说明 latent 空间记忆相比像素空间记忆有显著优势

## 相关概念

- [[视频扩散模型]]
- [[Latent Spatial Memory]]
- [[RealEstate10K]]
