---
type: concept
aliases: [PTv2, Point Transformer V2, 点云 Transformer]
---

# Point Transformer

## 定义

**Point Transformer** 系列是面向 3D 点云的 [[Transformer]] 架构，核心创新是 **向量自注意力 (vector self-attention)**：注意力权重不再是 query-key 的标量内积，而是逐通道的向量，使得每个特征通道能独立学习"邻居对自身的影响强度"。常用于点云分类、分割、场景理解，也是 [[Object-Centric Representation|物体中心]] 3D 表征中常见的特征提取骨干。

## 向量注意力公式

对中心点 $p_i$ 与其 k-近邻 $p_j \in \mathcal{N}(p_i)$：

$$
y_i = \sum_{j \in \mathcal{N}(p_i)} \rho\!\left(\gamma(\varphi(x_i) - \psi(x_j) + \delta_{ij})\right) \odot (\alpha(x_j) + \delta_{ij})
$$

- $\varphi, \psi, \alpha$: 线性投影
- $\delta_{ij}$: 相对位置编码（基于 $p_j - p_i$）
- $\gamma$: 小型 MLP，输出形状与特征同维（vector attention）
- $\rho$: softmax / normalization
- $\odot$: 元素逐通道相乘

## 与标准自注意力对比

| 维度 | Scalar Attention | Vector Attention (PT) |
|------|-----------------|-----------------------|
| 注意力形状 | 标量权重 | 与特征同维向量 |
| 表达力 | 较弱（共享权重） | 强（通道独立） |
| 内存 | 较小 | 较大 |
| 适用 | 序列/图像 | 3D 点云、稀疏几何 |

## PTv2 改进点

- **Grouped Vector Attention**: 把通道分组，组内共享注意力，降低显存
- **Position Encoding Multiplier**: 改进相对位置编码
- **Partition-based Pooling**: 网格划分的多尺度下采样

## 应用场景

1. 3D 物体分类 (ModelNet)
2. 室内场景分割 (S3DIS, ScanNet)
3. 物体中心 3D 世界模型（MRO-GWM 用 PTv2-based 向量注意力处理物体锚点）
4. SLAM / 重建后处理

## 代表工作

- **Point Transformer** (Zhao 2021)
- **Point Transformer V2 (PTv2)** (Wu 2022)
- **MRO-GWM** (2026): 在物体锚点 / Gaussian 中心上应用 PTv2 风格的稀疏向量注意力

## 关联概念

- [[Transformer]]
- [[自注意力]]
- [[Spatio-Temporal Attention]]
- [[3D Gaussian Splatting]]
