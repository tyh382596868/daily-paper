---
type: concept
aliases: [Language Expert, 语言专家]
---

# Language Expert

## 定义

Language Expert 是 [[WLA]] 等多专家具身基础模型中负责**生成自然语言子任务**的 head。基于 backbone 输出（含历史观测、当前指令、记忆），自回归解码出当前步要执行的子任务文本 $\mathcal{S}_t$，作为下游 World / Action Expert 的语义条件。

## 核心要点

1. **AR 文本解码**: next-token CE 损失，权重小（[[WLA]] 中 $\beta = 0.005$）
2. **语义子目标**: 输出形如 "pick up the red cup" 的可解释中间表征
3. **Chain-of-Thought 风格**: 让策略"先想后做"，类似 [[Chain-of-Thought|CoT]]
4. **依赖标注**: 需要人工子任务标注数据，规模化是挑战

## 代表工作

- [[WLA]]: Language Expert + World Expert + Action Expert 三专家架构
- [[CoT-VLA]]: 语言推理 + 动作的早期 VLA 变体
- [[Embodied CoT]]: 通用具身推理框架

## 相关概念

- [[Chain-of-Thought]]
- [[Embodied Reasoning]]
- [[Vision-Language-Action Model]]
