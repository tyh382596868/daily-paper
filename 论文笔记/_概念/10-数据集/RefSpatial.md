---
type: concept
aliases: [RefSpatial]
---

# RefSpatial

## 定义

一个用于空间引用与定位的小规模数据集（约 250 样本），用于训练模型理解带空间关系的自然语言指令（如 "the cup on the left of the bowl"）。

## 核心要点

1. **规模小**: ~250 个样本，主要用作精细对齐而非大规模训练。
2. **空间关系**: 包含 left/right/on/under/between 等空间介词。
3. **引用消歧**: 给定图像 + 语言描述 → 定位被引用的物体。
4. **VLA 预训练辅助**: AffordanceVLA Stage I 使用，强化空间 grounding 能力。

## 代表工作

- [[AffordanceVLA]]: Stage I 预训练数据

## 相关概念

- [[AGD20K]]
- [[PRISM]]
- [[Affordance Forecasting]]
- [[VLM]]
