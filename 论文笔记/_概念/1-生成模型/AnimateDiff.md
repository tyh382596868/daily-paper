---
type: concept
aliases: [AnimateDiff]
---

# AnimateDiff

## 定义
基于 Stable Diffusion 的视频生成框架。通过在预训练 T2I 模型中插入"运动模块（Motion Module）"，无需重训主干即可将静态图像生成器升级为视频生成器。

## 核心思路
- 冻结 SD 主干（保留图像先验和 LoRA 兼容性）
- 在 UNet 每个 block 后插入跨帧 self-attention（Motion Module）
- 仅训练 Motion Module，让其学到通用运动先验
- 推理时把 Motion Module 拼回任何 SD 变体上，生成视频

## 在 WM/VDM 中的位置
- 是早期"图像扩散 → 视频扩散"路径的代表
- 常被作为 [[CogVideoX]] / Sora 之前的 baseline
- 在 [[YoCausal]] 等 WM 评测工作中作为评测对象

## 相关概念
- [[Diffusion Model]]、[[CogVideoX]]、[[DiT]]
