---
type: concept
aliases: [LTX-2, LTX Video 2, LTX-2.3]
---

# LTX-2

## 定义

Lightricks 开源的视频扩散生成模型系列（LTX-Video 的第二代），提供高压缩比时空 VAE（8× 时间下采样）和大尺寸 DiT 主干（17B 级精修器）；在 SANA-WM 中被用作 VAE 编码器与第二阶段精修器底座。

## 核心要点

1. 时空 VAE：把高分辨率视频压成紧凑 latent，时间维度下采样 8×
2. DiT 主干为 [[Flow Matching]] 范式训练
3. 自带 short-video 精修器（LTX-2.3），但直接用于长视频会破坏一致性
4. 开源，适合做长视频生成基座或精修器底座

## 代表工作

- [[SANA-WM]]: 用 LTX-2 的 VAE；精修器用 17B LTX-2 + [[LoRA]] rank-384 + [[Truncated-σ Refiner]] 自适应训练

## 相关概念

- [[扩散变换器]]
- [[Flow Matching]]
- [[Truncated-σ Refiner]]
- [[LDM]]
