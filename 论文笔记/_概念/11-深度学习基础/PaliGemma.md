---
type: concept
aliases: [PaliGemma-3B]
---

# PaliGemma

## 定义

Google 发布的开源 [[VLM]]，由 [[SigLIP]]-So400m 视觉编码器与 [[Gemma]]-2B 语言模型组成，总参数约 3B。专为下游 fine-tune 设计，作为多个 [[VLA]] 模型的 backbone。

## 核心要点

1. **视觉塔**：SigLIP-So400m，27 层 Transformer，14×14 patch，hidden=1152，FF=4304；
2. **语言塔**：Gemma-2B，18 层 decoder，hidden=2048，FF=16384；
3. 训练数据多样、prompt 形式统一，泛化好；
4. 在 [[π₀]]、[[Pi05|π₀.₅]]、[[AR-VLA]] 等 VLA 工作中作为冻结 backbone 广泛使用。

## 代表工作

- Beyer et al., "PaliGemma: A versatile 3B VLM for transfer" (2024)
- [[π₀]] / [[Pi05|π₀.₅]] / [[AR-VLA]]：作为 VLM backbone

## 相关概念

- [[SigLIP]]
- [[Gemma]]
- [[VLM]]
- [[VLA]]
