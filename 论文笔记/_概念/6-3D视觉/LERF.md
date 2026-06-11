---
type: concept
aliases: [Language Embedded Radiance Fields, 语言嵌入辐射场]
---

# LERF

## 定义
"Language Embedded Radiance Fields"，在 NeRF/3DGS 等 3D 表示中额外蒸馏 [[CLIP]] 语言-图像特征，使得可以用自然语言在 3D 场景中查询/定位物体。

## 核心要点
1. 多尺度从 CLIP 蒸馏 patch 特征到 3D 表示。
2. 只对**已观测**内容附语义，不生成未观测区域。
3. 是"语义 3D 表示"这一支的代表，启发了 3D-Belief 的语义头设计。

## 代表工作
- [[3D-Belief]]: 语义头借鉴 LERF 式 CLIP 蒸馏，但进一步对想象出的未观测区域也赋语义；在相关工作中把 LERF/[[ConceptGraph]] 归为"只覆盖已观测"的局限

## 相关概念
- [[CLIP]]
- [[3D Gaussian Splatting]]
- [[ConceptGraph]]
