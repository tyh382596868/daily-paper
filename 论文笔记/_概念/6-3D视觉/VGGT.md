---
type: concept
aliases: [Visual Geometry Grounded Transformer]
---

# VGGT

## 定义
"Visual Geometry Grounded Transformer"，一个前馈式 Transformer，从一张到多张图像直接预测相机位姿、深度图、点图等 3D 几何属性，无需逐场景优化。

## 核心要点
1. pose-free：不需要输入相机位姿，模型自己估计。
2. 单次前向即可重建，速度快，常被用作"把 2D 想象 lift 到 3D"的模块。
3. 本身只做几何重建，不含生成式想象，也不含语义。

## 代表工作
- [[3D-Belief]]: 用作对比基线（DFoT-VGGT / NWM-VGGT：先视频生成再用 VGGT lift 到 3D），并在 Table 4 能力对比中作为"只有场景记忆"的代表

## 相关概念
- [[多视图 Transformer]]
- [[3D Gaussian Splatting]]
- [[新视角合成]]
