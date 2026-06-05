---
type: concept
aliases: [WAN Video, Wan2.1]
---

# WAN

## 定义

**WAN** 是阿里巴巴开源的大规模视频扩散模型系列（Wan2.1 / Wan2.2），基于 [[Diffusion Transformer]] 架构，是当前开源视频生成的主流基座之一，常被作为 [[Cosmos3]] 等新模型的对比基线。

## 核心要点

1. **架构**: DiT + [[3D Causal VAE]] + [[Flow Matching]]
2. **规模**: 14B / A14B 等多档
3. **能力**: 文生视频 / 图生视频 / 视频续写
4. **vs Cosmos 3**: WAN 是纯生成模型，Cosmos 3 在 WAN 类生成基础上叠加 VLM / VLA

## 关联

- [[Video Diffusion Model]]
- [[Diffusion Transformer]]
- [[Cosmos3]]
