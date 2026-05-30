---
type: concept
aliases: [VDM, 视频扩散模型, Video Diffusion Model]
---

# Video Diffusion Model (VDM)

## 定义

将 [[Diffusion Model|扩散模型]] 扩展到视频域的生成模型类。输入文本/图像 prompt，输出时空一致的视频帧序列，是当前主流的视频生成范式。

## 数学形式

训练目标（与 image diffusion 同形）：

$$
\mathcal{L}_{\mathrm{denoise}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
$$

其中 $x_t$ 是带噪视频，$c$ 是 text/image 条件。

## 核心要点

1. 主流架构：[[DiT]] 或 U-Net + 时间维度扩展
2. 训练数据规模庞大，对计算资源要求高
3. 评测维度：视觉质量（[[FVD]]）、文本对齐（[[CLIPScore]]）、世界知识（[[YoCausal]]）
4. 与 [[World Model|世界模型]] 的关系是当前的争论焦点

## 代表工作

- [[AnimateDiff]]、[[CogVideoX]]、[[Mochi-1]]、[[HunyuanVideo]]、[[Wan2.1]]、[[Wan2.2]]、[[LTX-Video]]
- [[YoCausal]]: 用因果性探针评测其"世界知识"

## 相关概念

- [[Diffusion Model]]
- [[Flow Matching]]
- [[World Model]]
- [[VBench]]
