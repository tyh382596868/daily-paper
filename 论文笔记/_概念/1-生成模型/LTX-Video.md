---
type: concept
aliases: [LTX-Video-2B, LTX-Video-13B, LTX]
---

# LTX-Video

## 定义

Lightricks 开源的高效 [[Video Diffusion Model|视频扩散模型]] 系列，主打**实时推理**（移动端友好）和小参数量。

## 核心要点

1. 2B / 13B 两档参数量
2. 强调推理速度，单 GPU 实时
3. 在 [[YoCausal]] 评测中表现反常：RSI 全场最高（2B 版本 58.86%），但 CCI 接近 0（−0.20%）— 暗示其感知到了时间方向（可能靠数据分布），但完全没学到因果区分

## 代表工作

- LTX-Video (Lightricks, 2024)
- [[YoCausal]]: 一个有趣的 case — "RSI 强但 CCI 弱"的代表

## 相关概念

- [[Video Diffusion Model]]
- [[DiT]]
- [[Reverse Surprise Index]]
