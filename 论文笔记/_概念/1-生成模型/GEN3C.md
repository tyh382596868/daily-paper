---
type: concept
aliases: [GEN3C, 3D-Cache Video Generation]
---

# GEN3C

## 定义
一种带 3D 缓存(cache)的可控视频生成方法：维护一个显式的 3D 点云缓存以保证跨帧/跨视角一致性，再用视频扩散模型基于该缓存渲染出精确相机控制的视频。

## 核心要点
1. 显式 3D（点云缓存）+ 2D 想象（扩散）+ 序贯更新；但 3D 主要用于一致性/重建，不是不确定性感知的多假设想象，也无语义。
2. 偏重视频生成的视觉与几何一致性。

## 代表工作
- [[3D-Belief]]: 在 RealEstate10K 上作对比（Table 5），并在 Table 4 中作为"有场景记忆、有 2D 想象、无 3D 想象、有序贯、无语义"的代表

## 相关概念
- [[视频扩散模型]]
- [[3D Gaussian Splatting]]
- [[ViewCrafter]]
