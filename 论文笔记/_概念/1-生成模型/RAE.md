---
type: concept
aliases: [RAE, Representation Autoencoder, 表征自编码器]
---

# RAE

## 定义

RAE（Representation Autoencoder）泛指一类"在预训练表征（如 [[DINO]] token）空间直接做生成扩散/流匹配"的自编码器世界模型范式。在 [[RLA-WM]] 论文里作为基线：直接把 DINO token $s_{t+h}$ 当作扩散目标，不做残差压缩。

## 核心要点

1. **直接在高维特征空间生成**：不像 [[残差潜在动作|RLA]] 那样压缩残差，RAE 把约 100 万维的 DINO token 当生成目标。
2. **维度灾难**：DINO token 维度比 Stable Diffusion VAE 潜变量（约 1.6 万维）还高一个量级，导致生成质量差、FLOPs 巨大。
3. **实验表现**：在 [[RLA-WM]] 论文的 ManiSkill/IWS 预测质量对比里，RAE 是表现最差的基线（ManiSkill LPIPS 0.324、IWS LPIPS 0.550），印证"不能直接在高维特征空间做扩散"。

## 代表工作

- 作为基线出现在 [[RLA-WM]]（Zhang et al., 2026）

## 相关概念

- [[DINO]]
- [[Flow Matching]]
- [[残差潜在动作]]
- [[世界模型]]
