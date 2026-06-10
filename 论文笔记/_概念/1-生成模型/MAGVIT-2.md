---
type: concept
aliases: [MAGVIT2, Masked Generative Video Transformer v2]
---

# MAGVIT-2

## 定义
Google 提出的离散视频 tokenizer，通过查找表（codebook）将视频帧压缩为离散潜在令牌，是高效视频生成和自回归视频建模的基础组件。

## 数学形式

$$
z = \text{Enc}(v) \in \mathbb{Z}^{T \times \frac{H}{\alpha} \times \frac{W}{\alpha}}
$$

其中 $\alpha$ 为空间降采样因子，$v$ 为原始视频帧序列，$z$ 为离散令牌索引。

## 核心要点
1. 使用向量量化（VQ）将连续特征映射到有限 codebook，支持自回归 Transformer 对视频令牌建模
2. 相比 MAGVIT v1，v2 在重建质量和压缩效率上均有提升
3. 空间维度降采样 $\alpha$ 倍，大幅减少序列长度，适合长视频自回归生成

## 代表工作
- [[RoboScape]]: 将 MAGVIT-2 用于机器人视频的离散分词，支持物理感知世界模型的自回归生成

## 相关概念
- [[VQ-VAE]]
- [[自回归视频生成]]
- [[离散潜在空间]]
