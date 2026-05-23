---
type: concept
aliases: [3DGS, Gaussian Splatting, 3D 高斯泼溅]
---

# 3D Gaussian Splatting

## 定义

一种显式 3D 场景表示方法：用大量各向异性 3D 高斯（位置、协方差、不透明度、SH 颜色）表示场景，通过可微分泼溅渲染（splatting）实现实时高质量新视角合成。

## 数学形式

每个高斯定义为：

$$
G_i(x) = \exp\!\left(-\frac{1}{2}(x - \mu_i)^\top \Sigma_i^{-1}(x - \mu_i)\right)
$$

渲染时按深度排序 $\alpha$-blend：

$$
C(p) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

## 核心要点

1. 显式表示，渲染速度可达 100+ FPS，远超 [[NeRF]] 的隐式 MLP
2. 训练阶段通过梯度分裂/剪枝/复制自适应控制高斯数量
3. 支持快速 4D 扩展（4DGS）、动态场景、几何编辑
4. 在世界模型/视频数据增广中常作为"渲染器"生成多样轨迹

## 代表工作

- [[SANA-WM]]: 用 FCGS 拟合 DL3DV 场景，渲染 40 条多样相机轨迹做训练数据增广
- [[GaussianDream]]: 用前馈高斯重建 + 未来预测给 VLA 注入 3D 监督，每帧 $256\times256 = 65536$ 个 surface-attached 高斯

## 相关概念

- [[相机投影]]
- [[NeRF]]
- [[Pi3X]]
