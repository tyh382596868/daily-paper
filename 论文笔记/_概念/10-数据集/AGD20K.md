---
type: concept
aliases: [AGD20K, Affordance Grounding Dataset]
---

# AGD20K

## 定义

一个面向 affordance grounding 的大规模标注数据集，含约 20K 张图像，标注了 "在哪个物体的哪个区域可执行某动作" 的可供性区域，常用于训练可供性预测模型。

## 核心要点

1. **20K 量级**: 图像数 ~20K，标注 affordance 区域（point/region）。
2. **动词类别**: 含数十种常见交互动词（pick、push、open 等）。
3. **任务**: affordance grounding——给定图像 + 动词 → 输出区域 / 热力图。
4. **用作 VLA 预训练**: AffordanceVLA Stage I 主要预训练数据之一。

## 代表工作

- [[AffordanceVLA]]: Stage I 可供性预训练

## 相关概念

- [[Affordance]]
- [[Affordance Forecasting]]
- [[RefSpatial]]
- [[PRISM]]
