---
type: concept
aliases: [ViT-Tiny, ViT-T]
---

# ViT-Tiny

## 定义

Vision Transformer 系列中最小的标准配置，由 DeiT (Touvron et al., 2021) 提出：12 层、3 头注意力、hidden=192，约 5M 参数。常见 patch 大小 16；在小数据 / 资源受限场景被广泛使用。

## 数学形式

标准 [[ViT]] 流程：图像切 patch → 线性投影 → 加 [CLS] token + position embedding → $L$ 层 [[Transformer]] block → 取 [CLS] 输出做下游头。

## 核心要点

1. **极轻量**：~5M 参数即可在 ImageNet 上单 GPU 训练
2. **与 ResNet-18 同量级**：常作为消融基线对比 [[Transformer]] vs CNN backbone
3. **应用场景**：移动端、自监督预训练消融、嵌入式机器人 encoder

## 代表工作

- DeiT (Touvron et al., 2021): 引入 ViT-Tiny / Small 配置 + 蒸馏
- [[LeWM]]: 用 ViT-Tiny 作为 encoder（patch=14、hidden=192）

## 相关概念

- [[Transformer]]
- [[JEPA]]
