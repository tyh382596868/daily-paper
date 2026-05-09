---
type: concept
aliases: [Joint Self-Attention, 联合自注意力, 跨流自注意力]
---

# Joint Self-Attention

## 定义
Joint Self-Attention 是一种多模态融合机制，将来自不同模态或流的 token 序列拼接后统一执行自注意力计算，实现自由的跨模态信息交互，同时各模态保留独立的投影参数。

## 数学形式

设有 $K$ 个模态流 $\{\mathbf{X}_1, \ldots, \mathbf{X}_K\}$，拼接后联合注意力：

$$
\mathbf{Z} = \text{Concat}(\mathbf{X}_1, \ldots, \mathbf{X}_K)
$$

$$
[\mathbf{X}'_1, \ldots, \mathbf{X}'_K] = \text{SelfAttention}(\mathbf{Z})
$$

各模态可使用独立的 Q/K/V 投影（modality-specific）但共享注意力计算。

## 核心要点

1. **完全跨模态交互**: 所有模态 token 都可互相注意，信息流动不受限制
2. **模态保护**: 通过独立投影层保留各模态专用表示，避免简单拼接导致的模态混淆
3. **计算效率**: 比逐对交叉注意力更高效，一次前向传播完成所有跨模态交互
4. **与模态特定架构对比**: 相比每个模态独立 Transformer 再做 cross-attention，Joint Self-Attention 更参数高效

## 代表工作

- [[RLDX-1]]: 在 MSAT 中用 Joint Self-Attention 融合认知流、动作流和物理流
- [[DiT]]: 类似思路用于扩散 Transformer 中的条件融合

## 相关概念

- [[MSAT]]
- [[Transformer]]
- [[VLA]]
