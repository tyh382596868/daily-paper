---
type: concept
aliases: [Joint-Aligned Latent Action, 联合对齐潜在动作]
---

# JALA（Joint-Aligned Latent Action）

## 定义

一种面向 VLA 大规模预训练的潜在动作学习框架，通过联合对齐从野外（in-the-wild）视频提取的潜在动作表示，实现可扩展的无标注视频数据利用。发表于 CVPR 2026（BeingBeyond 团队）。

## 核心要点

1. **面向 OXE 扩展预训练**: 将潜在动作表示与 Open X-Embodiment 数据对齐，扩展 VLA 预训练规模
2. **与 LARA 的区别**: JALA 侧重跨具身/跨数据集的预训练扩展，LARA 侧重 LAM 与 VLA 联合优化的双向正则化
3. **同期工作**: 与 [[LARA]] 均在 2025-2026 年关注 LAM+VLA 协同训练问题

## 代表工作

- 本身即代表工作（CVPR 2026，arXiv:2602.21736）

## 相关概念

- [[LAM]]
- [[LARA]]
- [[VLA]]
- [[OXE]]
