---
type: concept
aliases: [Chain of Thought, 思维链]
---

# CoT

## 定义
一种提示/训练技术，让语言模型在给出最终答案之前生成中间推理步骤，显著提升复杂推理任务的准确率。

## 核心要点
1. Zero-shot CoT：在 prompt 末尾加 "Let's think step by step"
2. Few-shot CoT：在示例中包含推理链
3. 在 VLA 中扩展为 Embodied Reasoning（ER），输出动作前先推理场景状态

## 代表工作
- Wei et al., 2022: CoT 原始论文
- [[MolmoAct2]]: 将 CoT 推理引入 VLA 动作预测

## 相关概念
- [[VLA]]
- [[VLM]]
