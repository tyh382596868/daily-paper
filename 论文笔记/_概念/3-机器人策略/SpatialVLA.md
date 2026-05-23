---
type: concept
aliases: [SpatialVLA]
---

# SpatialVLA

## 定义

把**显式 3D / 空间表征**注入 [[VLA]] 主干的一类方法，相比纯 2D RGB VLA 在涉及空间推理的任务上表现更好；常作为空间记忆型 VLA 的 baseline。

## 核心要点

1. 用 3D 点云 / 深度 / 位姿做空间编码注入 [[VLM]] 主干
2. 适用于桌面操作、空间关系推理任务
3. 在 [[OOV (Out-of-Vision)|OOV]] 任务上仍不如显式空间记忆（[[SOMA]]）——空间编码是逐帧的，没有跨视角累积

## 代表工作

- 在 [[SOMA]] 等论文中作为真机 OOV PnP 基线对比

## 相关概念

- [[VLA]]
- [[3D Gaussian Splatting]]
- [[VGGT]]
- [[SOMA]]
