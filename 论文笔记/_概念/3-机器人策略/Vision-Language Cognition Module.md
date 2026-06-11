---
type: concept
aliases: [VLC Module, 视觉语言认知模块]
---

# Vision-Language Cognition Module（视觉语言认知模块）

## 定义

MemoryVLA 系列中的感知编码阶段：使用双流视觉编码器（DINOv2 + SigLIP）和大语言模型（LLaMA-7B）将当前 RGB 观测与语言指令联合编码为结构化的感知 token（低层细节）和认知 token（高层语义推理），两者共同构成工作记忆。

## 核心要点

1. **双流视觉**：DINOv2 提取感知细节，SigLIP 提供视觉-语言对齐
2. **LLM 认知推理**：利用 LLaMA-7B 的常识先验生成语义丰富的认知 token
3. **工作记忆构建**：$W_t = [P_t; C_t]$，感知 token + 认知 token 联合形成工作记忆

## 代表工作

- [[MemoryVLA]]（ICLR 2026）
- [[MemoryVLA++]]（arXiv 2606.09827）

## 相关概念

- [[DINOv2]]
- [[SigLIP]]
- [[VLA]]
- [[Perceptual-Cognitive Memory Bank]]
