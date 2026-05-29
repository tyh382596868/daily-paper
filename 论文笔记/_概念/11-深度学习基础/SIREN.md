---
type: concept
aliases: [Sinusoidal Representation Network, Sinusoidal Representation Networks, 正弦表示网络]
---

# SIREN

## 定义

SIREN（Sinusoidal Representation Networks）是 Sitzmann 等人于 2020 年提出的一类全连接 MLP，使用**周期性 $\sin$ 激活**取代 ReLU/GELU，可对图像、3D 形状、视频、波场等**连续信号**做 $C^\infty$ 光滑的[[Neural Implicit Representation|隐式表示]]，并允许直接对输入坐标求**解析高阶导数**。

## 数学形式

$$
\mathbf{h}^{(\ell)} = \sin\!\big(\omega_0 (\mathbf{W}^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)})\big)
$$

其中 $\omega_0$ 为频率因子（典型值 30），首层权重以 $\mathcal{U}(-1/n_{in}, 1/n_{in})$ 初始化、后续层以 $\mathcal{U}(-\sqrt{6/n}/\omega_0, \sqrt{6/n}/\omega_0)$ 初始化以稳定训练。

## 核心要点

1. **$C^\infty$ 光滑**：$\sin$ 的任意阶导数仍是 $\sin/\cos$，可直接用 [[Autograd]] 求得解析速度、加速度、jerk；
2. **频谱偏置友好**：相比 ReLU MLP 的低频偏置，SIREN 能拟合高频细节；
3. **与 Hypernetwork 天然兼容**：每层权重可被外部网络调制频率/相位（[[Grouped Hyper-Modulation]]）；
4. **初始化敏感**：$\omega_0$ 选错会导致训练发散，需按论文标准初始化。

## 代表工作

- **SIREN 原论文** (Sitzmann et al., 2020)：图像 / 视频 / 3D 形状 / 泊松方程
- [[NIAF]]：把 SIREN 用作 VLA 的**动作场**，由 [[MLLM]] 调制频率/相位

## 相关概念

- [[Neural Implicit Representation]]
- [[Hypernetwork]]
- [[Grouped Hyper-Modulation]]
- [[Autograd]]
