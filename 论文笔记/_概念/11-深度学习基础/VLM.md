---
type: concept
aliases: [VLM, Vision-Language Model, 视觉语言模型]
---

# VLM

## 定义

Vision-Language Model，统一处理图像（或视频）和文本的多模态大模型，通常由视觉编码器 ([[ViT]] / [[SigLIP]] 等) + 语言模型 ([[Transformer]] 解码器) 组成，输入图像 + 文本指令，输出文本。

## 核心要点

1. 架构：vision tower (frozen 或可微调) + projector + LLM
2. 训练：图文对比预训练 → 视觉指令微调
3. 是 [[VLA]] 的直接 backbone（在 VLM 后接动作头即可）
4. 也常作机器人系统中 S2 高层规划器（见 [[System 1 / System 2 双系统架构]]）

## 代表工作

- [[Qwen3-VL]]、CLIP、SigLIP、Chameleon、LLaVA 等

## 相关概念

- [[ViT]]
- [[Transformer]]
- [[VLA]]
- [[Qwen3-VL]]
- [[CLIP]]
- [[SigLIP]]
