---
type: concept
aliases: [DINOv2]
---

# DINOv2

## 定义

DINOv2 (Oquab et al., 2024) 是 Meta AI 在大规模 curated 图像数据集（~142M 张）上自监督预训练的 [[Transformer]] 视觉 encoder，沿用 [[DINO]] 的自蒸馏 + iBOT mask token 目标。在密集预测、深度估计、分类、检索等任务上无需微调即达到强基线，是机器人视觉的事实标准 backbone 之一。

## 数学形式

总体目标 = self-distillation + iBOT mask + KoLeo + Sinkhorn-Knopp 中心化（多目标组合，详见原论文）。

## 核心要点

1. **大规模 + 高质量**：自动 curated 数据 + 多目标自监督
2. **强通用 backbone**：常被用作下游冻结 encoder
3. **机器人应用**：[[DINO-WM]]、OSVI-WM 等世界模型直接冻结 DINOv2 作 encoder
4. **缺点**：~1B 参数级别，每帧产出 ~200 tokens，下游 [[Transformer]] 推理重

## 代表工作

- Oquab et al., 2024: DINOv2 原始论文
- [[DINO-WM]]: 冻结 DINOv2 + 训 predictor 的世界模型
- [[LeWM]]: 用端到端 [[ViT-Tiny]] (~5M) 在多任务上追平甚至超过 DINOv2-based 方案

## 相关概念

- [[DINO]]
- [[Transformer]]
- [[DINO-WM]]
