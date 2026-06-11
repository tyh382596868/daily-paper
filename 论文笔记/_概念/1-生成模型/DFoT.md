---
type: concept
aliases: [Diffusion Forcing Transformer, History-Guided Video Diffusion]
---

# DFoT

## 定义
"Diffusion Forcing Transformer"（History-Guided Video Diffusion），一种用历史帧引导的视频扩散模型：对序列中不同帧施加独立噪声水平，支持以可变长度历史为条件做长 horizon、可控相机运动的视频/新视角生成。

## 核心要点
1. 强 2D 想象 + 序贯（自回归）生成，但停留在像素空间，没有显式 3D 信念。
2. 长 horizon 下易出现语义漂移、几何/位姿漂移。
3. 常被"先 DFoT 生成视频，再用 [[VGGT]] lift 到 3D"的方式当作 3D 基线（DFoT-VGGT）。

## 代表工作
- [[3D-Belief]]: 主要对比基线（2D 视觉质量、3D-CORE、导航三处均对比），并在 Table 4 中作为"无场景记忆、有 2D 想象、无 3D 想象、有序贯、无语义"的代表

## 相关概念
- [[视频扩散模型]]
- [[NWM]]
- [[扩散模型]]
