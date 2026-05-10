---
type: concept
aliases: [Molmo2, Molmo 2]
---

# Molmo2

## 定义

Allen Institute for AI（AI2）开发的多模态视觉-语言模型，具备像素级指点、视频理解和多图对应能力，是 MolmoAct2 机器人系统的基础骨干网络。

## 核心要点

1. 开放权重 VLM，支持图像 QA、视频 QA、像素级指点等多种多模态任务
2. 专项具身推理版本 Molmo2-ER 在 3.3M 具身样本上额外训练，在 13 个具身推理 benchmark 平均 63.8%
3. 采用"专化-复习"两阶段训练防止灾难性遗忘

## 代表工作

- [[MolmoAct2]]: 以 Molmo2-ER 为 VLA 骨干，通过逐层 KV-Cache 连接流匹配动作专家

## 相关概念

- [[VLA]]
- [[具身推理]]
- [[Flow Matching]]
