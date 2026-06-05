---
type: concept
aliases: [SANA]
---

# SANA

## 定义

SANA 是一种高效的 [[Diffusion Transformer|DiT]] 变体，主打线性注意力 + 深度压缩 VAE，在保持图像质量的同时显著提速。在 [[WLA]] 中作为 600M 参数的 [[World Expert]] backbone 用于未来子目标帧生成。

## 核心要点

1. **线性注意力**: 把扩散 Transformer 的 $O(n^2)$ 注意力替换为线性变体
2. **深度压缩 VAE**: 更大的下采样率，减少 token 数量
3. **轻量参数**: 几百 M 量级即可达到 SOTA 图像质量
4. **多模态条件**: 支持文本、图像等多种条件输入

## 代表工作

- [[WLA]]: 用 SANA-600M 作为 World Expert
- 原论文: NVIDIA 提出的高效 DiT

## 相关概念

- [[Diffusion Transformer]]
- [[VAE]]
- [[Flow Matching]]
