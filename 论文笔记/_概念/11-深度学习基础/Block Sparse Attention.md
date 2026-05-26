---
type: concept
aliases: [Block Sparse Attention, 块稀疏注意力, BSA]
---

# Block Sparse Attention

## 定义

Block Sparse Attention 把 attention 矩阵划分成**固定大小的块**，只对预先定义的"非零块集合"做计算，跳过其余块。常与"半因果"（semi-causal）掩码配合，在保持必要 token 交互的前提下让计算复杂度从 $O(N^2)$ 降为接近 $O(N)$。

## 数学形式

给定 token 数 $N$、块大小 $B$，把 attention 矩阵分成 $\lceil N/B \rceil^2$ 个 $B \times B$ 块。设非零块集合 $\mathcal{S}$：

$$
\text{Attn}(Q, K, V)_{ij} =
\begin{cases}
\text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d}} \right) V_j, & \text{若块} (i, j) \in \mathcal{S} \\
0, & \text{否则}
\end{cases}
$$

实际复杂度 $\propto |\mathcal{S}| \cdot B^2$。

## 核心要点

1. **块粒度**: 不在 token 粒度上做稀疏（GPU 不友好），而在块粒度，使 kernel 高效
2. **结构化稀疏**: 通常采用 chunk-local + 跨 chunk 邻域 + 全局 sink 等组合
3. **训练加速**: 在长序列任务上相对 [[FlashAttention-2]] 通常带来 1.5-3× 加速
4. **比 sliding window 更灵活**: 可以同时容纳"局部 + 全局 + 跨 chunk 位置对应"等多种 pattern

## 代表工作

- [[X-Foresight]]: 半因果块稀疏 attention，长视野 (21 s) 训练相对 FlashAttention-2 加速 1.59×
- BigBird / Longformer: 早期块稀疏 + 全局 token 的范式

## 相关概念

- [[FlashAttention-2]]
- [[自注意力]]
- [[滑动窗口注意力]]
