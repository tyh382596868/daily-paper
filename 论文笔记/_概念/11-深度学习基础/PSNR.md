---
type: concept
aliases: [PSNR, 峰值信噪比, Peak Signal-to-Noise Ratio]
---

# PSNR

## 定义
PSNR（峰值信噪比，Peak Signal-to-Noise Ratio）是衡量重建/生成图像与参考图像逐像素差异的指标，由均方误差换算为分贝（dB），数值越高表示重建越接近参考。

## 数学形式

$$
\text{PSNR}=10\cdot\log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

其中 $\text{MAX}$ 为像素最大值，$\text{MSE}$ 为与参考图的均方误差。

## 核心要点
1. 越高越好，常与 [[结构相似性|SSIM]]、[[感知图像相似度|LPIPS]] 一起报告。
2. 纯像素级指标，对感知质量不敏感（与人眼判断可能不一致），故需配合感知指标。
3. 常用于图像/视频重建、生成、回放一致性评估。

## 代表工作
- [[CoME]]: 在 Memory Maze、RealEstate10K 上用 PSNR 评估重建质量

## 相关概念
- [[结构相似性]]
- [[感知图像相似度]]
