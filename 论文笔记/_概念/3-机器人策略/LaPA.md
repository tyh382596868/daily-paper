---
type: concept
aliases: [Latent Action Pretraining]
---

# LaPA

## 定义

Latent Action Pretraining（LAPA）：用 [[VQ-VAE]]-based 潜在动作模型从无标签视频中自监督学习 "state→latent action" 先验，下游用少量真实动作微调即可映射到关节空间。

## 核心要点

1. **无动作标签预训练**: 极大减少对昂贵机器人数据的依赖。
2. **潜在动作空间**: 通过 VQ-VAE 离散化得到通用动作表示。
3. 属于 [[Cascaded WAM|Cascaded WAM]] 的 Implicit / Latent Representation 子类。

## 代表工作

- [[WAM-Survey]] 综述中作为 Cascaded-Implicit 代表。
- [[villa-X]] 在 LaPA 思路上进一步引入 proprioception forward dynamics。

## 相关概念

- [[残差潜在动作]]
- [[World Action Model]]
- [[VLA]]
