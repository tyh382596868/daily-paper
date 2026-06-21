---
type: concept
aliases: [LLaMA2, Llama 2, llama-2]
---

# LLaMA-2

## 定义
Meta AI 发布的第二代开源大语言模型系列（7B/13B/70B），在 LLaMA 基础上提升了训练数据量（2T tokens）和安全对齐（RLHF），被广泛用作 VLA 等多模态模型的语言主干。

## 核心要点
1. **参数规模**: 7B / 13B / 70B，7B 版本是 VLA 应用最常用规格
2. **开源许可**: 相对宽松的商用许可，促进了学术和工业应用
3. **在 VLA 中的应用**: 作为语言主干接收多模态 token，通过 LoRA 等参数高效微调适配机器人任务

## 代表工作
- [[OpenVLA]]: 采用 LLaMA-2 7B 作为语言骨干
- [[PearlVLA]]: 基于 OpenVLA-7B（LLaMA-2 7B）构建

## 相关概念
- [[LoRA]]
- [[OpenVLA]]
- [[VLA]]
