---
type: concept
aliases: [K400, Kinetics 400]
---

# Kinetics-400

## 定义

DeepMind 发布的大规模人类动作视频识别数据集（2017），包含 400 个动作类别、约 30 万段 YouTube 短视频，每段 ~10 秒。

## 核心要点

1. 视频识别的"ImageNet"，大量 SOTA 视频模型用其预训练
2. 涵盖日常动作、运动、人际互动等 400 类
3. 在 [[YoCausal]] 中作为 Human Action 子集（400 视频，3 秒切片）评测 [[Video Diffusion Model|VDM]] 的人类动作因果理解

## 代表工作

- Kay et al. "The Kinetics Human Action Video Dataset" (arXiv 2017)
- 大量动作识别 / VDM 工作的训练/评测源

## 相关概念

- [[Moments in Time]]
- [[VBench]]
