---
type: concept
aliases: [SSIM, Structural Similarity, 结构相似性指数]
---

# SSIM (Structural Similarity Index)

## 定义

一种结合**亮度 + 对比度 + 结构**三方面的图像质量评价指标，对人眼感知的"结构一致性"敏感。取值范围 $[-1, 1]$，**越接近 1 越相似**，1 表示两图完全一致。

## 数学形式

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

其中 $\mu, \sigma, \sigma_{xy}$ 分别是局部均值、方差、协方差，$C_1, C_2$ 是稳定项。

## 核心要点

1. **越高越好**（范围 [-1, 1]，1 最佳）
2. **结构敏感**：对图像的局部结构变化敏感
3. **窗口计算**：通常用 $11 \times 11$ 高斯窗口
4. **MS-SSIM**：多尺度变体能进一步提升评价能力

## 代表工作

- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (TIP 2004)
- [[GAF]] 用 SSIM 评估 GS 重建质量，相对 [[ManiGaussian]] 提升 +0.3864

## 相关概念

- [[PSNR]]
- [[LPIPS]]
- [[3D Gaussian Splatting]]
- [[结构相似性]]
