---
type: concept
aliases: [RynnBrain, Rynn Brain]
---

# RynnBrain

## 定义

RynnBrain 是 [[WLA]] 使用的 2.1B 参数 AR Transformer backbone，初始化自 VLM 预训练权重，作为多模态上下文（文本指令、历史/当前图像、机器人状态、记忆）的统一编码器。

## 核心要点

1. **2.1B 参数**: 相对轻量，便于实时推理
2. **AR Transformer**: 因果注意力，可同时处理 next-token 解码（语言子任务）和 [[Meta Query|meta query]] 聚合（多专家共享）
3. **VLM 初始化**: 沿用视觉-语言预训练表征，省去从零训练的成本
4. **多模态统一接口**: 视觉 token、文本 token、状态 token、记忆 token 共用同一 backbone

## 代表工作

- [[WLA]]: 首次系统使用 RynnBrain 作为 backbone

## 相关概念

- [[Vision-Language Model]]
- [[Meta Query]]
- [[Action Expert]]
