---
type: concept
aliases: [潜在空间记忆, 潜空间记忆, latent cache]
---

# Latent Spatial Memory

## 定义

将视频世界模型的场景记忆从 RGB 像素空间迁移到 VAE latent 空间的 3D 空间记忆表示，以 $(\mathbf{p}_i, \mathbf{f}_i)$ 配对形式存储世界坐标和 latent 特征。

## 数学形式

$$
\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}, \quad \mathbf{p}_i \in \mathbb{R}^3,\ \mathbf{f}_i \in \mathbb{R}^C
$$

与 RGB 点云表示对比：

$$
\mathcal{M}_{\text{rgb}} = \{(\mathbf{p}_i, \mathbf{c}_i)\}, \quad \mathbf{c}_i \in [0,1]^3
$$

## 核心要点

1. **消除 rasterise-and-encode 往返**: 读出的 latent 特征直接与 backbone 同分布，无需 VAE 编码桥接
2. **三阶段流水线**: 初始化（深度引导反投影）→ 读出（z-buffer 投影）→ 更新（新帧融合）
3. **动态过滤**: 排除运动物体和天空区域防止几何污染
4. **效率优势**: 相比 RGB 点云方案，显存降低约 55×，速度提升约 10.57×

## 代表工作

- [[Mirage]]: 提出 Latent Spatial Memory 的原始论文，用于无边界视频世界模型

## 相关概念

- [[VAE]]
- [[视频扩散模型]]
- [[ControlNet]]
- [[z-buffer]]
- [[深度引导反投影]]
