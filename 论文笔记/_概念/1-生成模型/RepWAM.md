---
type: concept
aliases: [Representation World Action Model]
---

# RepWAM

## 定义
表示中心的世界动作模型，提出 RepViTok —— 联合优化视觉保真度和表示一致性的 visual-action tokenizer。

## 数学形式
$$\mathcal{L}_\text{RepViTok} = \mathcal{L}_\text{recon} + \lambda \mathcal{L}_\text{repr-align}$$

## 核心要点
1. 现有 WAM 继承重建导向 tokenizer，导致表示质量与动作对齐脱节
2. RepViTok 同时优化重建和表示对齐
3. 用 IDM 从 unlabeled video 中恢复 action token
4. Fudan + HKU 提出，在 RoboTwin 上验证

## 代表工作
- [[RepWAM]]: arXiv 2606.13674

## 相关概念
- [[WAM]]
- [[IDM]]
- [[RoboTwin]]
