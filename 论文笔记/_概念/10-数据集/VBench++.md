---
type: concept
aliases: [VBench Plus Plus]
---

# VBench++

## 定义

VBench++ 是 VBench 的扩展版本，在原有视频质量评测维度基础上增加了更多细粒度评测指标，用于全面评估文本到视频生成模型的质量。

## 核心要点

1. **扩展维度**：在 [[VBench]] 的 16 个维度基础上进一步细化，涵盖更多视觉质量子维度。
2. **局限性**：仅关注单轮视频生成的外观质量，不评测物理合规性、3D 几何一致性或长时序交互。
3. **对比 WorldOlympiad**：[[WorldOlympiad]] 覆盖 VBench++ 无法评测的物理/几何/交互三个维度。

## 代表工作

- [[VBench]]: 前置工作
- [[WorldOlympiad]]: 对比基准，扩展至世界模型评测

## 相关概念

- [[VBench]]
- [[FVD]]
- [[视频生成]]
