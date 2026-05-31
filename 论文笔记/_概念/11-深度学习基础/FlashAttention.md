---
type: concept
aliases: [FlashAttention, Flash Attention]
---

# FlashAttention

## 定义

一种 IO-aware 的精确注意力算法（Tri Dao et al., 2022），通过分块（tiling）+ 重计算（recomputation）把注意力的中间矩阵 $QK^\top$ 保留在 GPU SRAM 中，避免与 HBM 的反复读写，从而显著降低 wall-clock 时间和显存峰值。

## 数学形式

标准注意力：
$$
O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

FlashAttention 用 online softmax（Milakov & Gimelshein, 2018）把行最大值 $m$ 和归一化常数 $\ell$ 增量更新：

$$
m^{(j)} = \max(m^{(j-1)}, \tilde m_j),\quad
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}}\ell^{(j-1)} + e^{\tilde m_j - m^{(j)}}\tilde\ell_j
$$

按块计算输出累加，无需具现化 $N \times N$ 的 attention matrix。

## 核心要点

1. **IO-aware**：性能瓶颈是 HBM ↔ SRAM 的内存访问，而非 FLOPs。分块让中间结果留在 SRAM。
2. **精确注意力**：与朴素 attention 数学等价，没有近似（区别于 sparse / linear attention）。
3. **线性显存**：显存随序列长度线性增长 $O(N)$，而非朴素的 $O(N^2)$。
4. **kernel 级实现**：用 CUDA / Triton 写成融合 kernel，框架层面通过 `torch.nn.functional.scaled_dot_product_attention` 自动调用。
5. **后续版本**：FlashAttention-2 优化 work partitioning；FlashAttention-3 利用 Hopper 的异步 + FP8。

## 代表工作

- FlashAttention (NeurIPS 2022): 提出原始算法。
- FlashAttention-2 (2023): 提升 GPU 利用率。
- FlashAttention-3 (2024): 针对 H100 的 warp-level 异步流水。

## 相关概念

- [[注意力机制]]
- [[Transformer]]
