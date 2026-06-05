---
type: concept
aliases: [理解专家]
---

# Understanding Expert

## 定义

AffordanceVLA 中负责"建立视觉感知与语言意图细粒度对齐"的 MoT 专家，基于预训练 VLM 的先验权重，输出指令感知的语义表征 $h^{und}_t$。

## 核心要点

1. **语义对齐**: 接收图像 + 语言 token，输出对齐的多模态表征，作为下游可供性和动作的上下文。
2. **复用 VLM 先验**: 利用预训练 VLM 的权重，避免从头训练视觉语言对齐。
3. **UAA 起点**: 在 [[UAA 注意力机制]] 中位于上游，其 KV 被 Affordance 与 Action 专家 query，但反向被屏蔽。
4. **Stage I 冻结**, Stage II 解冻参与联合训练。

## 代表工作

- [[AffordanceVLA]]

## 相关概念

- [[Affordance Generation Expert]]
- [[Action Expert]]
- [[MoT]]
- [[VLM]]
- [[UAA 注意力机制]]
