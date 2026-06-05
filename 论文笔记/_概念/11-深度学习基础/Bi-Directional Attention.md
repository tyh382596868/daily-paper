---
type: concept
aliases: [双向注意力, Bidirectional Attention, Full Attention]
---

# Bi-Directional Attention

## 定义

**Bi-Directional Attention**（双向 / 全注意力）允许每个 token attend 到序列中所有其他 token，不施加因果 mask。适用于编码任务（BERT）、扩散模型去噪（DiT）以及 [[Cosmos3]] 的扩散生成塔。

## 核心要点

1. **无 mask**: $M_{ij} = 0, \forall i, j$
2. **适合非自回归**: 扩散模型一次性预测整段噪声 latent，需要 token 间全局互访
3. **训练成本高**: 复杂度仍是 $O(n^2)$
4. **vs Causal**: 扩散塔用双向、AR 塔用因果，[[MoT|MoT 架构]] 同一层中两种 mask 共存

## 代表工作

- [[Cosmos3]]: 扩散生成塔使用 bidirectional attention
- [[Diffusion Model]] 系列的 DiT backbone
- [[BERT]] 类编码器

## 关联

- [[Causal Self-Attention]]: 对立
- [[JointSelfAttention]]: 多模态联合双向注意力
- [[Diffusion Model]]
