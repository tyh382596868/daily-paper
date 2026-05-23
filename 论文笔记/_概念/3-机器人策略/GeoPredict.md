---
type: concept
aliases: [GeoPredict]
---

# GeoPredict

## 定义

一种基于纯几何预测（不含 RGB 渲染监督）的机器人操作策略基线。在 [[GaussianDream]] 论文中作为 RoboCasa Human-50 上的强对比基线出现，验证仅深度 / 几何预测监督也能提升 VLA 表现，但加上 RGB 渲染会更好。

## 核心要点

1. 不依赖可微分渲染，只用几何相关监督（深度 / 3D 流）
2. 在 [[RoboCasa]] Human-50 上达到 52.4% 平均成功率
3. 在 Doors/Drawers / Others 类任务上表现强于 [[GaussianDream]]（75.1 vs 65.2; 62.4 vs 52.0），但 pick-and-place 弱（22.7 vs 43.8）
4. 暗示 RGB 渲染监督对**精细定位**贡献更大，而粗略 affordance 类任务可能仅几何就够

## 代表工作

- [[GaussianDream]]: 作为 RoboCasa 上的对比基线

## 相关概念

- [[GaussianDream]]
- [[3D Gaussian Splatting]]
- [[VLA]]
