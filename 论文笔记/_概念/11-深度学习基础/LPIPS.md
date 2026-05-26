---
type: concept
aliases: [LPIPS, Learned Perceptual Image Patch Similarity, 感知图像块相似度]
---

# LPIPS (Learned Perceptual Image Patch Similarity)

## 定义

一种基于预训练 CNN 深度特征的图像感知相似度度量：把两张图分别送入冻结的 AlexNet / VGG / SqueezeNet，对各层特征做归一化后逐层加权 L2 距离求和。比传统 [[PSNR]] / [[SSIM]] 更贴近人眼对图像差异的感知判断。**数值越低越相似**。

## 数学形式

$$
\text{LPIPS}(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}_l^{hw} - \hat{y}_{0,l}^{hw}) \|_2^2
$$

其中 $\hat{y}_l$ 是第 $l$ 层归一化后的特征，$w_l$ 是学习到的层权重。

## 核心要点

1. **越低越好**（与 PSNR/SSIM 相反）
2. **预训练特征**：通常用冻结的 AlexNet/VGG
3. **感知友好**：对纹理、对比度变化的判断接近人类
4. **常用于 NVS / GS**：是 [[3D Gaussian Splatting|3DGS]]、NeRF 类工作的标配指标

## 代表工作

- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
- [[GAF]] 用 LPIPS 监督高斯渲染（公式 5），重建上 LPIPS 比 [[ManiGaussian]] 改善 -0.5574

## 相关概念

- [[PSNR]]
- [[SSIM]]
- [[3D Gaussian Splatting]]
- [[感知图像相似度]]
