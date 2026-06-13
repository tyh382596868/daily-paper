---
type: concept
aliases: [Depth Anything, DepthAnything v3]
---

# DepthAnything

## 定义

一系列开源单目深度估计基础模型，以大规模预训练实现强泛化的相对/度量深度预测，被广泛用于下游 3D 重建和场景记忆初始化任务。

## 核心要点

1. **版本演进**: DepthAnything v1 → v2 → v3（DepthAnything 3），精度和泛化性逐步提升
2. **相对 vs 度量深度**: 基础版输出无尺度相对深度；度量版（Metric）可输出绝对深度
3. **在世界模型中的作用**: 为 [[深度引导反投影]] 提供深度图，Mirage 默认使用 DepthAnything 3
4. **深度源鲁棒性**: Mirage 消融显示对 DepthAnything 3 / MapAnything / UniDepth 不敏感（差异 <1.5%）

## 代表工作

- [[Mirage]]: 使用 DepthAnything 3 作为深度估计器，初始化 [[Latent Spatial Memory]]

## 相关概念

- [[深度引导反投影]]
- [[Latent Spatial Memory]]
