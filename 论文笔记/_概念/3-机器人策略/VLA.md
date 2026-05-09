---
type: concept
aliases: [Vision-Language-Action Model, 视觉语言动作模型]
---

# VLA（Vision-Language-Action Model）

## 定义

VLA 是将视觉感知、语言理解与机器人动作预测统一到单一模型中的端到端框架，以预训练视觉-语言模型（VLM）为骨干，直接输出机器人控制指令。

## 核心要点

1. **统一建模**：图像 + 语言 + 动作在同一模型中联合处理
2. **大规模预训练**：利用互联网规模的视觉-语言数据提升语义理解
3. **动作解码**：可用自回归 token、扩散策略或流匹配等方式生成动作

## 代表工作

- [[OpenVLA]]：早期开源 VLA 代表
- [[π₀]]：基于流匹配的高性能 VLA
- [[π₀.₅]]：π₀ 的增强版，开箱即用性强
- [[GR00T N1.7]]：NVIDIA 的双系统 VLA
- [[MolmoAct2]]：完全开源的具身推理 VLA

## 相关概念

- [[Action Chunking]]
- [[Flow Matching]]
- [[具身推理]]
