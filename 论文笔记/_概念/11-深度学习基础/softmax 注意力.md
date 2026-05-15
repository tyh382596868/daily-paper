---
type: concept
aliases: [Softmax Attention, 标准注意力, Vanilla Attention]
---

# softmax 注意力

## 定义
[[Transformer]] 中的标准注意力机制：用 softmax 对 $QK^\top$ 做归一化后加权聚合 value。

## 数学形式

$$
\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

## 核心要点

1. **精确表达力**: 能任意"路由"注意力到全局任何 token，是 [[线性注意力]] 难以替代的属性
2. **复杂度 $O(N^2)$**: 长序列下 KV cache 与计算都随 token 数二次增长
3. **长视频瓶颈**: 60s 视频（~961 潜帧 × S 空间 token）下纯 softmax 在 H100 会 OOM
4. **混合架构常用**: 与 [[线性注意力]] / [[Frame-wise Gated DeltaNet|GDN]] 交错使用，少量 softmax 层即可保留精确召回
5. **优化**: FlashAttention、Sliding Window、attention sink 等

## 代表工作
- 原始 Transformer
- [[SANA-WM]]: 在 20 层中插入 5 个 softmax 块（位置 3, 7, 11, 15, 19）
- 几乎所有 LLM / DiT

## 相关概念
- [[Transformer]]
- [[线性注意力]]
- [[Frame-wise Gated DeltaNet]]
- [[chunk-causal]]
