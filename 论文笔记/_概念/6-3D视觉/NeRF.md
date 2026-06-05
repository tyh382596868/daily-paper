---
type: concept
aliases: [NeRF, Neural Radiance Field, 神经辐射场]
---

# NeRF（Neural Radiance Field）

## 定义

一种用 MLP 隐式参数化 3D 场景的方法：网络输入 3D 位置 + 视角方向，输出该点的密度和颜色，通过体渲染从多视角图像监督训练。

## 数学形式

MLP $F_\Theta : (x, d) \mapsto (\sigma, c)$，沿光线 $r(t) = o + td$ 的体渲染：
$$
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) \, dt
$$
其中 $T(t) = \exp(-\int_{t_n}^t \sigma(r(s)) ds)$。

## 核心要点

1. **隐式表征**: MLP 参数 = 场景表征，无显式几何。
2. **多视角监督**: 用多视角图像训练，无需 3D ground truth。
3. **新颖视图合成**: 质量高，但训练/渲染慢。
4. **被 [[3DGS]] 取代**: 在速度上 3DGS 优势明显，NeRF 在研究中仍是基线。
5. **机器人应用**: 用作场景表征、可供性 grounding、动力学模型。

## 代表工作

- NeRF 原文（Mildenhall et al., 2020）
- Instant-NGP（加速）
- Block-NeRF / Mip-NeRF

## 相关概念

- [[3DGS]]
- [[Volume Rendering]]
- [[Implicit Representation]]
