---
type: concept
aliases: [Gemma-2B, Gemma LLM]
---

# Gemma

## 定义

Google 开源的轻量 LLM 系列，2B / 7B 等多个尺寸。架构基于 Gemini 同源设计，采用 [[RoPE]]、[[RMSNorm]]、GeGLU 等现代组件，适合做 [[VLM]] 的语言塔。

## 核心要点

1. 2B 版本：18 层 decoder，hidden=2048，FF=16384；
2. 使用 [[RoPE]] 位置编码（[[Dynamic Temporal Re-anchoring]] 的基础）；
3. 在 [[PaliGemma]] 中作为语言塔；
4. 是 [[π₀]]、[[Pi05|π₀.₅]]、[[AR-VLA]] 等 VLA 间接依赖的语言模型。

## 代表工作

- Gemma Team, "Gemma: Open Models Based on Gemini Research and Technology" (2024)
- [[PaliGemma]]：将 Gemma 作为 VLM 的语言塔

## 相关概念

- [[PaliGemma]]
- [[RoPE]]
- [[RMSNorm]]
- [[VLM]]
