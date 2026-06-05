---
type: concept
aliases: [DVD, Direct Video Depth, DVD-DiT]
---

# DVD Depth

## 定义

基于 [[Diffusion Transformer]] 的视频深度估计模型，输出多帧一致的归一化视差。

## 核心要点

1. 在视差空间（$d = 1/(z+\epsilon)$）训练，逐 clip 用鲁棒分位数归一化
2. 损失包含 L1 + 空间梯度 + 时间梯度匹配
3. 可通过 [[LoRA]] 在机器人 rollout 上轻量微调
4. [[Dream-exe]] 把它作为 V2T 流水线的关键深度模块

## 代表工作
- [[Dream-exe]]
- DVD 原始论文

## 相关概念
- [[Depth Anything V2]]
- [[Video Diffusion Models]]
- [[LoRA]]
- [[Video-to-Trajectory]]
