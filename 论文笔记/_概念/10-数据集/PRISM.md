---
type: concept
aliases: [PRISM, Interaction-Aware Scene Data]
---

# PRISM

## 定义

一个面向 embodied VQA 的大规模交互感知数据集（约 412K 样本），含"针对场景物体的可交互性问题"，用于训练 VLM/VLA 理解物体的可操作性属性。

## 核心要点

1. **412K 样本**: VQA 形式，覆盖丰富物体与交互问题。
2. **交互感知**: 问题涉及 "can I push this?"、"is this graspable?" 等可供性导向问题。
3. **预训练**: 用于在 affordance grounding 之外补充更细粒度的语义理解。
4. **AffordanceVLA Stage I 主力**: 与 AGD20K、RefSpatial 共同构成 Stage I 数据。

## 代表工作

- [[AffordanceVLA]]: Stage I 数据

## 相关概念

- [[AGD20K]]
- [[RefSpatial]]
- [[VLM]]
- [[Embodied VQA]]
