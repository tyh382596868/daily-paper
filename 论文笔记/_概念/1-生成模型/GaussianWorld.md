---
type: concept
aliases: [Gaussian World Model]
---

# GaussianWorld

## 定义
基于 3D Gaussian Splatting 的自动驾驶场景世界模型，将场景表示为可渲染的 Gaussian 集合，支持新视角合成和场景动态预测。

## 核心要点
1. 用 [[3DGS]] 表示场景（显式几何 + 外观）
2. 对 Gaussian 参数做时序预测（运动、形变）
3. 可渲染任意视角，适合自动驾驶多摄像头评估

## 代表工作
- 用于 VISA 论文的 3D occupancy world model baseline

## 相关概念
- [[3DGS]]
- [[OccWorld]]
- [[高斯泼溅]]
