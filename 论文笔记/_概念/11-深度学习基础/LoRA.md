---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配]
---

# LoRA

## 定义
一种参数高效微调（PEFT）方法，通过在预训练权重旁并联低秩矩阵分解来近似全量微调，大幅减少可训练参数量。

## 数学形式
$$W' = W_0 + \Delta W = W_0 + BA$$
其中 $W_0 \in \mathbb{R}^{d \times k}$ 为冻结的预训练权重，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$。

## 核心要点
1. 仅训练 A、B 矩阵，预训练权重冻结
2. 推理时可将 $\Delta W$ 合并到原权重，无额外延迟
3. 秩 $r$ 通常取 4-64

## 代表工作
- Hu et al., 2022: LoRA 原始论文（NeurIPS）

## 相关概念
- [[MoE]]
- [[SVD]]
- [[VLA]]
