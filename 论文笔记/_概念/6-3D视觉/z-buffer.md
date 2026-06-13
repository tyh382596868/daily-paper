---
type: concept
aliases: [深度缓冲, 深度测试, depth buffer, depth testing]
---

# z-buffer

## 定义

计算机图形学中用于解决可见性（遮挡）问题的经典算法：为每个像素维护一个深度值缓冲，光栅化时只保留深度最小（最近）的片段。

## 数学形式

对每个屏幕像素 $(u,v)$，选取投影到该位置的所有 3D 点中深度最小的点：

$$
i^*(u,v) = \mathop{\mathrm{arg\,min}}_{i \in \Omega(u,v)} [\mathbf{E} \mathbf{p}_i]_z
$$

## 核心要点

1. **时间复杂度**: $O(N)$，对每个 3D 点做一次投影和比较
2. **空间局部性**: 逐像素独立操作，便于 GPU 并行
3. **在世界模型中的应用**: [[Mirage]] 将 z-buffer 扩展到 latent 分辨率，直接在 latent 格点上做深度测试，读出最近点的 latent 特征

## 代表工作

- [[Mirage]]: 在 latent-resolution 下用 z-buffer 实现 Latent Spatial Memory 的读出

## 相关概念

- [[Latent Spatial Memory]]
- [[深度引导反投影]]
- [[Pinhole Camera Model]]
