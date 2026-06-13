---
type: concept
aliases: [针孔相机模型, 小孔成像, pinhole model]
---

# Pinhole Camera Model

## 定义

计算机视觉中最基础的相机几何模型，通过内参矩阵 $K$ 和外参矩阵 $\mathbf{E}$ 描述 3D 世界坐标到 2D 图像坐标的线性投影关系。

## 数学形式

投影（3D → 2D）：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \mathbf{E} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

反投影（2D + 深度 → 3D），参见 [[深度引导反投影]]：

$$
\mathbf{p} = \mathbf{E}^{-1} \begin{bmatrix} d \cdot K^{-1}[u, v, 1]^\top \\ 1 \end{bmatrix}\bigg|_{1:3}
$$

## 核心要点

1. **内参 $K$**: 焦距 $f_x, f_y$ 和主点 $c_x, c_y$，描述像素坐标到归一化坐标的映射
2. **外参 $\mathbf{E}$**: 旋转 + 平移，描述世界坐标到相机坐标的刚体变换
3. **latent 分辨率内参**: 对于 latent 空间操作，内参需缩放：$K^\ell = \mathrm{diag}(w/W, h/H, 1) K$

## 代表工作

- [[Mirage]]: 在 latent 分辨率下使用针孔模型做 [[深度引导反投影]]，构建 [[Latent Spatial Memory]]

## 相关概念

- [[深度引导反投影]]
- [[z-buffer]]
