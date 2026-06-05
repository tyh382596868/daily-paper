---
type: concept
aliases: [3D Gaussian Splatting, 3DGS, Gaussian Splatting]
---

# 3D Gaussian Splatting（3DGS）

## 定义

一种基于显式 3D 高斯点云表征 + 可微 splatting 渲染的新颖视图合成方法，速度快、质量高，逐渐取代 NeRF 成为 3D 场景表征主流方案。

## 核心要点

1. **显式表征**: 用一组 3D 高斯（位置、协方差、颜色、不透明度）表示场景，区别于 NeRF 的隐式 MLP。
2. **可微 Splatting**: 高斯投影到 2D 屏幕做 alpha-blend，可微，支持端到端训练。
3. **实时渲染**: 训练快（分钟级），渲染快（>100 FPS），适合机器人/AR。
4. **可编辑**: 高斯点云可被直接编辑、移除、变形，区别 NeRF。
5. **应用**: 场景重建、4D 动态、机器人感知、可供性预测中可提供 3D 几何先验。

## 代表工作

- 3DGS 原文（Kerbl et al., 2023）
- Dynamic 3DGS / 4DGS / Splatter Image
- [[GAF|Gaussian Action Field]]: 把 3DGS 用于机器人动作场

## 相关概念

- [[NeRF]]
- [[4DGS]]
- [[Gaussian Action Field]]
- [[Point Cloud]]
