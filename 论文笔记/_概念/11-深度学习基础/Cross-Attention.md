---
type: concept
aliases: [Cross-Attention, 跨注意力, Cross Attention]
---

# Cross-Attention

## 定义

注意力的非自注意力变体——query 来自一个序列，key/value 来自**另一个**序列，用于让一组 token 主动从另一组 token 中"检索"信息。

## 数学形式

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

与 [[Self-Attention|自注意力]] 的区别在于 $Q = X_a W_Q$ 而 $K, V = X_b W_K, X_b W_V$，$X_a \neq X_b$。

## 核心要点

1. **不对称信息流**: 只从 source（$X_b$）流向 target（$X_a$），常用于编码器-解码器结构
2. **典型场景**: 语言 token 检索视觉 token、记忆库检索、扩散模型条件注入
3. **在机器人中**: 常作为"语言 → 空间记忆"的检索接口
4. 计算复杂度 $O(L_a \cdot L_b)$，不像 self-attention 是平方关系

## 代表工作

- [[SOMA]]: VLM token 作 query，3D 空间记忆作 key/value，做情境化检索
- [[MemoryVLA]]: 历史 token 作 KV，当前 token 作 Q
- 经典 Transformer 解码器中的 encoder-decoder attention

## 相关概念

- [[Self-Attention]]
- [[Transformer]]
- [[DiT]]
- [[VLM]]
