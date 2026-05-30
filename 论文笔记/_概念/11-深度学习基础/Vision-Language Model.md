---
type: concept
aliases: [VLM, 视觉语言模型, Vision-Language Model]
---

# Vision-Language Model (VLM)

## 定义

同时处理视觉输入（图像/视频）和语言输入/输出的多模态大模型。代表如 GPT-4V、Gemini、Claude、Qwen-VL、InternVL 等。

## 核心要点

1. 架构常用 vision encoder + LLM + cross-modal connector
2. 应用：图像理解、视频问答、多模态推理、[[VLM-as-Judge|VLM 评判]]
3. 在 [[YoCausal]] 中，Gemini 3.0 Pro 作为因果分层 judge，与人工 Kendall $\tau=0.7613$
4. 自身可能继承训练偏置，作为 judge 时需要做一致性校验

## 代表工作

- GPT-4V、Gemini 系列、Claude Vision
- [[InternVL]]、[[Florence-2]]、Qwen-VL
- [[YoCausal]]: 用 VLM 做因果/非因果分层

## 相关概念

- [[VLM-as-Judge]]
- [[VLM]]
- [[CLIP]]
