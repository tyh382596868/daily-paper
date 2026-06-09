---
type: concept
aliases: [LLaMA, LLaMA 7B, llama-7b, Meta LLaMA]
---

# LLaMA-7B

## 定义

Meta AI 发布的开源大语言模型系列中的 7B 参数版本（LLaMA = Large Language Model Meta AI）：基于 [[Transformer]] decoder-only 架构，在万亿 token 数据上训练，以高效的计算-性能比著称，广泛作为 VLA、具身智能等下游任务的语言骨干。

## 核心要点

1. **7B 参数**：在语言理解、推理能力与计算效率间取得较好平衡
2. **开源生态**：权重公开，大量工作在其上进行指令微调（如 Alpaca、Vicuna）
3. **在 VLA 中的作用**：作为语言指令编码器和常识推理引擎，输出认知 token

## 代表工作

- [[MemoryVLA]]（ICLR 2026）: 使用 LLaMA-7B 作为语言推理骨干
- [[OpenVLA]]：基于 LLaMA-2 的 VLA 模型

## 相关概念

- [[VLA]]
- [[Vision-Language Cognition Module]]
- [[SigLIP]]
