---
type: concept
aliases: [3D Causal VAE, 3D 因果 VAE, Causal 3D VAE]
---

# 3D Causal VAE

## 定义

3D Causal VAE 是一种把视频沿时间轴做**因果编码**的 [[VAE]] 变体：编码器看到第 $t$ 帧时仅依赖 $\le t$ 的帧，时间维上严格因果。常用作视频 [[扩散模型]] / 视频 [[DiT]] 的 latent backbone，将像素视频压缩到低维 4D 张量 $[C, T, H, W]$ 上做扩散。

## 核心要点

1. **时间因果**: 卷积或 transformer block 在时间维上用因果 mask，允许逐 chunk 编码 / 推理
2. **3D**: 同时在空间 $H \times W$ 和时间 $T$ 维做下采样（典型 8×8 空间 + 4 倍时间）
3. **作为 latent backbone**: 视频 DiT / Rectified Flow 视频生成器通常都在它的 latent 上工作
4. **代价小**: 一旦训完，视频 diffusion 训练只在低维 latent 上做，省下数十倍算力

## 代表工作

- [[X-World]]: 用 3D Causal VAE + DiT 做 7 相机环视照片级渲染
- [[X-Foresight]]: Vision Renderer 基于 X-World，因此也依赖 3D Causal VAE
- [[Wan2.1]] / [[Wan2.2-TI2V]]: 视频生成的 latent backbone

## 相关概念

- [[VAE]]
- [[DiT]]
- [[视频扩散模型]]
- [[Rectified Flow]]
