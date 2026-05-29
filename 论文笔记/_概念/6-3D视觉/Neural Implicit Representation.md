---
type: concept
aliases: [INR, Implicit Neural Representation, 神经隐式表示, Coordinate-based MLP]
---

# Neural Implicit Representation

## 定义

Neural Implicit Representation（INR）指用一个**坐标输入 → 信号输出**的 MLP 隐式地表达连续信号 $f: \mathbb{R}^n \to \mathbb{R}^m$，如图像 $(x, y) \to (r, g, b)$、NeRF $(x, y, z, \theta, \phi) \to (\sigma, c)$、SDF $(x, y, z) \to d$、动作场 $\tau \to a$ 等。

## 数学形式

$$
f_\theta: \mathbb{R}^n \to \mathbb{R}^m, \quad (x_1, \ldots, x_n) \mapsto f_\theta(x_1, \ldots, x_n)
$$

由 $\theta$（MLP 权重）隐式编码整段信号。

## 核心要点

1. **连续与无分辨率限制**：相对像素 / voxel 网格，INR 不绑定离散网格；
2. **频谱偏置问题**：朴素 ReLU MLP 偏低频，需借助 [[Positional Encoding]] 或 [[SIREN]] 才能拟合高频细节；
3. **与超网络结合**：[[Hypernetwork]] 给 INR 提供条件化能力；
4. **应用面广**：图像 / 视频 / NeRF / SDF / 物理仿真 / 动作场（[[NIAF]]）。

## 代表工作

- **NeRF** (Mildenhall et al., 2020)：5D 辐射场
- **SIREN** (Sitzmann et al., 2020)：$\sin$ 激活，$C^\infty$ 表示
- **DeepSDF** (Park et al., 2019)：用 MLP 表示 SDF
- [[NIAF]]：把 INR 思想搬到动作域

## 相关概念

- [[SIREN]]
- [[Hypernetwork]]
- [[Neural Implicit Action Field]]
