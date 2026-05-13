---
type: concept
aliases: [Neural Radiance Field, 神经辐射场]
---

# NeRF

## 定义
用一个 MLP 把空间坐标 $(x,y,z)$ 与视角方向 $(\theta,\phi)$ 映射到颜色与体密度，通过体渲染积分合成图像的隐式 3D 场景表示。

## 数学形式
$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt,\quad T(t)=\exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))ds\right)$$

## 核心要点
1. 隐式、连续，质量高但训练/渲染慢（每像素需多次 MLP 查询 + 沿光线积分）。
2. [[3D Gaussian Splatting|3DGS]] 是其显式、快速的替代方案。
3. 衍生大量变体（Instant-NGP、Mip-NeRF 等）加速与抗锯齿。

## 代表工作
- [[3D-Belief]]: 选用 3DGS 而非 NeRF，理由是显式 + 渲染快 + 可附语义

## 相关概念
- [[3D Gaussian Splatting]]
- [[新视角合成]]
