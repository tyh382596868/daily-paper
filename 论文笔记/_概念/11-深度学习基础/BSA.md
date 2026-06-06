---
type: concept
aliases: [Block Sparse Attention, 块稀疏注意力]
---

# BSA (Block Sparse Attention)

## 定义
Block Sparse Attention 是一种把注意力矩阵按 token 块（block）划分，只计算稀疏选定块内/块间相似度的高效注意力机制，用来把长上下文（长视频帧序列、超长 token 序列）的 attention 复杂度从 $O(N^2)$ 降到接近线性。

## 数学形式
标准 attention 计算所有 query-key 对：
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

BSA 引入一个 block-level 的稀疏 mask $M \in \{0,1\}^{B \times B}$，其中 $B$ 是 block 数量。被选中的 block 内部和跨 block 才计算 softmax：
$$\mathrm{BSA}(Q,K,V)_b = \mathrm{softmax}\!\left(\frac{Q_b K_{S(b)}^\top}{\sqrt{d}}\right)V_{S(b)}, \quad S(b)=\{b'|M_{b,b'}=1\}$$

## 核心要点
1. **Block 划分**：把 token 序列按固定大小分块（如 64/128 token 一块），attention 在块粒度上做选择
2. **稀疏模式**：常见做法包括 local window、stride、global token 几种组合（类似 Longformer / Big Bird 的思路）
3. **硬件友好**：块结构天然契合 GPU tile，可以直接用 FlashAttention 类的 kernel 实现
4. **典型应用**：长视频生成、长文档建模、长上下文 LLM 推理（如 LongCat-Video 的长视频生成）

## 代表工作
- [[LongCat-Video]]：把 BSA 用在视频生成 DiT 主干，配合 video-continuation 训练做长视频
- 类似思路的更早工作：Sparse Transformer、Longformer、Big Bird

## 相关概念
- [[FlashAttention-2]]
- [[DiT]]
- [[JointSelfAttention]]
