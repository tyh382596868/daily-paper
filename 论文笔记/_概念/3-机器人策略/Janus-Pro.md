---
type: concept
aliases: [Janus Pro, JanusPro]
---

# Janus-Pro

## 定义

Janus-Pro 是 DeepSeek 提出的统一多模态理解与生成基础模型，采用解耦的视觉编码路径，分别处理理解任务与生成任务，在 VLA 领域被用作底座模型。

## 核心要点

1. **解耦视觉路径**: 将视觉理解（用于推理）和视觉生成（用于图像合成）的编码器分离，避免任务间干扰
2. **LLM 主干**: 基于 DeepSeek-LLM，提供强语言推理能力
3. **VLA 适配**: 在 LaST-HD 等机器人策略工作中，替换或扩展生成头为流匹配动作专家，成为视觉-语言-动作模型底座
4. **多专家设计**: 与 Mixture-of-Transformers（MoT）结合，分离推理专家（负责潜在推理）和动作专家（负责动作预测）

## 代表工作

- [[LaST-HD]]: 以 Janus-Pro 为底座，加入 MoT 架构（推理专家 + 动作专家），实现跨体态潜在对齐机器人策略

## 相关概念

- [[混合专家 Transformer]]
- [[VLA]]
- [[SigLIP]]
- [[流匹配（Flow Matching）|Flow Matching]]
