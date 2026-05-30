---
type: concept
aliases: [RSI, 反向惊讶指数, Reverse Surprise Index]
---

# Reverse Surprise Index (RSI)

## 定义

由 [[YoCausal]] 提出的 Level 1 指标，用 [[Denoising Loss|去噪损失]] 比较正向视频与时间反转视频，量化 [[Video Diffusion Model|视频扩散模型]] 是否感知到 [[Arrow of Time|时间方向]]。

## 数学形式

$$
\mathrm{RSI}(\mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{\mathcal{D}_i \in \mathcal{D}} \frac{1}{|\mathcal{D}_i|} \sum_{x_{ij} \in \mathcal{D}_i} \mathbb{1}\!\left[ \mathcal{L}_{\mathrm{denoise}}(\theta; x^r_{ij}) > \mathcal{L}_{\mathrm{denoise}}(\theta; x^f_{ij}) \right]
$$

取值范围 $[0, 1]$，0.5 = 随机猜测。

## 核心要点

1. 信号来自正反样本的去噪损失差异
2. 不需要任何额外标注或训练，黑盒探针
3. 对训练分布敏感 — 跨模型可比性需要谨慎
4. Human baseline ≈ 79.08%，最强 VDM (LTX-Video-2B) ≈ 58.86%

## 代表工作

- [[YoCausal]]: 首次提出，用于 13 个 VDM 的因果性评测

## 相关概念

- [[Causality Cognition Index]]
- [[Arrow of Time]]
- [[Denoising Loss]]
- [[Counterfactual]]
- [[World Model]]
