---
type: concept
aliases: [PhysicsIQ, Physics-IQ]
---

# Physics IQ

## 定义

Google DeepMind 发布的物理因果视频评测集（2024），专门测视频模型是否理解物理规律（重力、碰撞、流体、相变等）。

## 核心要点

1. ~100 段精心构造的强物理因果场景
2. 用于评测 [[Video Diffusion Model|VDM]] 是否能预测真实物理动力学
3. 在 [[YoCausal]] 中作为 Physics 子集（132 视频，5 秒切片），human RSI 高达 91.7% — 物理因果信号最强
4. 多数 VDM 在此子集与 human 差距最大，是 VDM 的物理认知薄弱区

## 代表工作

- Motamed et al. "Physics IQ" (Google DeepMind, 2024)
- [[YoCausal]]: 借用作为因果探针的物理子集

## 相关概念

- [[Moments in Time]]
- [[Kinetics-400]]
- [[VBench]]
