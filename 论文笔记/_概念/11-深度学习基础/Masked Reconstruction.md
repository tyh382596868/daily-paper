---
type: concept
aliases: [遮掩重建, masked reconstruction, MAE, Masked Autoencoder]
---

# Masked Reconstruction

## 定义

自监督预训练技术：随机遮掩输入（图像块、patch 或 token），让模型从可见部分重建被遮掩部分，从而学习丰富的特征表示，无需人工标注。

## 数学形式

给定输入 $x$，采样遮掩比例 $p \sim \mathcal{U}(0, p_{\max})$，遮掩 patch 集合 $\mathcal{M}$：

$$
\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|f_\theta(x_{\setminus \mathcal{M}})_i - x_i\|^2
$$

实践中常结合感知损失（[[LPIPS]]）提升视觉质量。

## 核心要点

1. MAE（Masked Autoencoder，He et al., 2022）将高遮掩比（如 75%）应用于 ViT，显著提升编码器表示质量
2. 可用于视觉分词器预训练（[[MMBench2]] 使用每帧随机 $p \sim \mathcal{U}(0, 0.9)$ 遮掩）
3. 高遮掩比迫使模型学习全局结构，低遮掩比偏向局部细节

## 代表工作

- [[MMBench2]]：用于视频[[视频分词器|分词器]]预训练
- MAE（He et al., 2022）：原始提出者，开创性地将高比例遮掩用于 ViT

## 相关概念

- [[视频分词器]]
- [[LPIPS]]
- [[世界模型]]
