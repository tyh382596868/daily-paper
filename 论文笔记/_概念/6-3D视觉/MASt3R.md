---
type: concept
aliases: [MASt3R, MASt3R 模型]
---

# MASt3R

## 定义

DUSt3R 系列的进化版：一个从两张未标定图像直接前馈预测稠密 3D 点云、匹配和相机参数的几何 foundation 模型。提供 3D-aware 的 ViT 特征，常被下游任务作为权重初始化使用，绕开传统的 SfM / 标定流程。

## 数学形式

$$
\text{MASt3R}: \{I_1, I_2\} \to \{P_1, P_2, M_{1 \leftrightarrow 2}\}
$$

输出每张图的点图 $P$ 与跨图匹配 $M$。

## 核心要点

1. **无需相机位姿**：直接从未标定图像预测度量 3D
2. **稠密点对应**：输出像素级跨图匹配
3. **强 3D 先验**：ViT 特征中编码了几何信息
4. **常作初始化**：被 [[GAF]]、[[VGGT]] 等工作用作 backbone 初始化

## 代表工作

- MASt3R / MASt3R-SfM
- [[GAF]] 把 MASt3R 权重作为 ViT 初始化

## 相关概念

- [[ViT]]
- [[VGGT]]
- [[Structure-from-Motion]]
- [[3D Gaussian Splatting]]
