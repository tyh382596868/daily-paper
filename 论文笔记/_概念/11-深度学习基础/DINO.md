---
type: concept
aliases: [DINO, Self-DIstillation with NO labels]
---

# DINO

## 定义
Meta AI 提出的自监督视觉表示学习方法，通过知识蒸馏（无标签）训练 ViT，生成具有语义分割特性的特征表示。

## 数学形式
$$\mathcal{L} = -\sum_x P_t(x) \log P_s(x)$$
其中 $P_t$ 为 EMA 更新的 teacher 输出（经中心化+sharpen），$P_s$ 为 student 输出。

## 核心要点
1. teacher-student 架构，teacher 用 EMA 更新
2. centering + sharpening 避免 collapse
3. DINOv2 进一步扩展到更大规模

## 代表工作
- Caron et al., 2021: DINO 原始论文

## 相关概念
- [[JEPA]]
- [[SigLIP]]
- [[EMA]]
