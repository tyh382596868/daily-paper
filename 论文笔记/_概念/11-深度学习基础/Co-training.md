---
type: concept
aliases: [协同训练, Co-Training, 联合训练]
---

# Co-training

## 定义
一种训练范式：在同一组模型参数上**交替/混合**地优化多个相关任务或多个数据源的损失，让模型在共享表征上同时获得多种能力。在 VLA 语境下常指**VLA 动作数据 + VLM 视觉-语言数据**混合训练 VLM backbone。

## 数学形式

总损失通常是加权之和：
$$
\mathcal{L}_{total} = \mathcal{L}_{vla} + \lambda\,\mathcal{L}_{vlm}
$$
两类数据可以**按比例采样**也可以**交替 step**。

## 核心要点
1. 防止 VLA 微调过程中 VLM 的**灾难性遗忘**（catastrophic forgetting）
2. 通过 VLM 任务（QA、3D reasoning）注入额外**结构化知识**到表征
3. 数据比例需调参，常见 1:0.1 到 1:1
4. 必须解决**Prompt 路径冲突**——不同任务的 prompt 模板会让模型走不同推理路径（参见 [[Prompt-Induced Reasoning Gap]]）

## 代表工作
- [[3DThinkVLA]]: VLA + 3D reasoning QA 协同训练
- [[CoT-VLA]]: explicit CoT + 动作协同
- [[PaliGemma]]: vision-language 协同预训练范式

## 相关概念
- [[Multi-Task Learning]]
- [[Knowledge Distillation]]
- [[Continual Learning]]
