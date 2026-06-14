---
type: concept
aliases: [RGB Point Cloud, 彩色点云, 带色点云]
---

# RGB 点云

## 定义

RGB 点云（RGB Point Cloud）是一种三维场景表示方式，将空间中每个点 $\mathbf{p}_i \in \mathbb{R}^3$ 与对应的 RGB 颜色值 $\mathbf{c}_i \in [0,1]^3$ 配对存储：

$$
\mathcal{M}_{\text{rgb}} = \{(\mathbf{p}_i, \mathbf{c}_i)\}, \quad \mathbf{p}_i \in \mathbb{R}^3,\ \mathbf{c}_i \in [0,1]^3
$$

## 数学形式

在视频世界模型中，RGB 点云作为空间记忆时，需通过**光栅化 + 重新编码**将其转化为 latent 条件：

$$
\hat{\mathbf{z}}^t = \mathcal{E}(\mathrm{Rasterise}(\mathcal{M}_{\text{rgb}}; \mathbf{E}^t, K^t))
$$

其中 $\mathrm{Rasterise}(\cdot)$ 将 3D 点云投影渲染到 2D 图像，$\mathcal{E}$ 为 VAE 编码器。

## 核心要点

1. **三维表示**：每个点包含世界坐标位置和 RGB 颜色，支持任意视角的渲染
2. **深度来源**：通常由单目深度估计（如 [[DepthAnything]]）+ 相机反投影生成，也可来自 RGBD 传感器
3. **渲染方式**：目标视角下通过光栅化（z-buffer 深度测试）投影到 2D 图像平面
4. **与 latent 特征的区别**：RGB 颜色仅包含像素级外观信息（3通道），而 latent 特征包含更丰富的语义信息（如 48 通道）

## 局限性

- **rasterise-and-encode 往返开销**：每次条件化都需要光栅化到像素图再 VAE 编码，计算复杂度为 $\Theta(\Phi_{\mathcal{E}}(H,W))$
- **显存占用高**：RGB 点云需要存储像素分辨率的颜色信息，显存随轨迹长度线性增长
- **语义容量有限**：3 通道颜色信息丢失了扩散模型 latent 中编码的高维语义

## 代表工作

- [[Voyager]]: 使用 RGB 点云作为视频世界模型的长期空间记忆
- [[Spatia]]: 同样基于 RGB 点云的空间一致性视频生成
- [[Mirage]]: 提出 [[Latent Spatial Memory]] 替代 RGB 点云，效率提升 55×

## 相关概念

- [[Latent Spatial Memory]]: 替代 RGB 点云的 latent 空间记忆表示
- [[深度引导反投影]]: 从深度图构建点云的技术
- [[z-buffer]]: 点云渲染时的深度测试机制
- [[VAE]]: RGB 点云到 latent 的桥接编码器
- [[点云]]: 更通用的 3D 点集表示概念
