---
type: concept
aliases: [ViTPose, Vision Transformer Pose Estimation]
---

# ViTPose

## 定义

基于 Vision Transformer 的 2D 人体姿态估计模型，通过预训练大规模 ViT 主干网络提取图像特征，输出全身关键点热图（heatmap），支持全身（包括手部、面部）关键点检测。

## 数学形式

$$
\hat{J}_{2D} = \text{ViTPose}(I) \in \mathbb{R}^{K \times 2}
$$

其中 $K$ 为关键点数量，$I$ 为输入图像。

## 核心要点

1. 使用 ViT 作为主干，相比 CNN 方法有更强的全局感知能力
2. 支持 body、wholebody（含手/面部）等多种关键点集合
3. 在 COCO-WholeBody 等数据集上取得 SOTA 性能
4. 常用于为后续 3D 姿态提升（HMR2、VIMO 等）提供 2D 输入

## 代表工作

- [[GRAIL]]: GEM-SMPL 中使用 ViTPose 提供全身 2D 关键点，作为 HOI 优化器的关键点损失输入

## 相关概念

- [[SMPL]]
- [[HMR2]]
- [[MPJPE]]
