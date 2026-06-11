---
type: concept
aliases: [ConceptGraphs, 开放词表 3D 场景图]
---

# ConceptGraph

## 定义
一种开放词表的 3D 场景表示：把 RGB-D 序列聚合成以物体为节点、关系为边的 3D 场景图，每个物体节点附带视觉-语言特征，支持语言查询与规划。

## 核心要点
1. 物体级、结构化、带语义，便于下游 LLM/VLM 规划。
2. 只表示已观测物体，不对未观测区域做生成式想象。

## 代表工作
- [[3D-Belief]]: 相关工作中把 ConceptGraph/[[LERF]] 等语义 3D 表示归为"聚焦已观测、缺少未观测区域想象"的一类

## 相关概念
- [[LERF]]
- [[CLIP]]
- [[3D Gaussian Splatting]]
