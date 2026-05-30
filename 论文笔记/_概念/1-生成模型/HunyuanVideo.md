---
type: concept
aliases: [Hunyuan-Video, 混元视频]
---

# HunyuanVideo

## 定义

腾讯混元团队开源的 13B 级 [[Video Diffusion Model|视频扩散模型]]（2024 年底），是开源 T2V 中规模较大的一档。

## 核心要点

1. 13B+ 参数，DiT 架构 + 3D VAE
2. 在文本对齐和视觉质量上对标闭源模型
3. 在 [[YoCausal]] 评测中：RSI 52.05%，CCI −0.29%（几乎为 0）— 时间方向感知不错但因果认知缺失

## 代表工作

- HunyuanVideo (Tencent, 2024)
- [[YoCausal]]: 评测对象之一

## 相关概念

- [[Video Diffusion Model]]
- [[DiT]]
- [[3D Causal VAE]]
