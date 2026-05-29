---
type: concept
aliases: [Florence2, Florence-2 Large]
---

# Florence-2

## 定义

Florence-2 是微软发布的轻量级多模态视觉基础模型，统一处理 caption、grounding、OCR、segmentation 等多种视觉-语言任务，模型规模有 Base (230M) 与 Large (770M) 两种。在机器人领域被广泛用作 [[VLA]] 的视觉-语言主干。

## 核心要点

1. **统一 prompt 化任务**：所有任务都通过特定 prompt token 触发；
2. **小而强**：Large 仅 770M，却能匹配主流 BLIP / Flamingo 级别效果；
3. **decoder 易接动作头**：seq2seq 结构方便附加动作生成模块；
4. **NIAF 默认主干**：[[NIAF]] 在 Florence-2 Large 上做主线对比。

## 代表工作

- **Florence-2** (Microsoft, 2024)
- [[NIAF]]：以 Florence-2 Large 为主线主干

## 相关概念

- [[MLLM]]
- [[Qwen3-VL]]
- [[VLA]]
- [[Pi05]]
