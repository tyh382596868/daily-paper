---
type: concept
aliases: [模态特化编码器, Per-Modality Encoder]
---

# Modality-Specific Encoder

## 定义

**Modality-Specific Encoder** 指多模态基座模型中为每种输入模态设计的独立前置编码器，把原始信号（图像像素、视频、音频波形、动作向量）压缩到与主干 Transformer 兼容的 token 表示。

## 核心要点

1. **视觉理解侧**: [[ViT]] 提取语义 token（保留全局信息）
2. **视觉/音频生成侧**: [[3D Causal VAE]] / 音频 VAE 压缩为连续 latent（保留细节）
3. **动作侧**: [[Embodiment Anchoring|embodiment-specific projection]]，每种机器人形态有专属线性层
4. **统一**: 编码后再走 [[3D mRoPE]] 共享位置编码

## 代表工作

- [[Cosmos3]]: 同时使用 ViT (理解) + 3D Causal VAE (生成) + Embodiment Projection (动作)
- [[Qwen3-VL]]: 视觉理解用 ViT

## 关联

- [[ViT]]
- [[3D Causal VAE]]
- [[Embodiment Anchoring]]
- [[3D mRoPE]]
