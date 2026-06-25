---
type: concept
aliases: [FlashAttention-2, Flash Attention 2, FA2]
---

# FlashAttention-2

## 定义

FlashAttention-2 是对原始 FlashAttention 的工程优化版本（Tri Dao, 2023），通过 **tile + IO-aware** 算法减少 GPU 显存读写，把 attention 计算速度相对朴素实现提升 2-4×，且数值上完全等价于 full attention（不引入稀疏近似）。

## 核心要点

1. **IO 感知**: 将 $QKV$ 矩阵分块加载到 SRAM，避免反复读写 HBM
2. **数值等价**: 与 full softmax attention 严格等价，不是稀疏化
3. **反向也优化**: forward + backward 都做了 tile 化
4. **vs Block Sparse Attention**: FA2 仍是 $O(N^2)$ 复杂度，对**真正长**的序列（如 X-Foresight 的 21 s × 7 相机）仍嫌慢，需要更稀疏的 [[Block Sparse Attention]]
5. **PyTorch / FlashAttention 库**: 已成为大模型训练标配

## 代表工作

- Tri Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- [[X-Foresight]] 作为 baseline 对照（Block Sparse Attention 比 FA2 加速 1.59×）
- [[Causal-rCM]]: 扩展 FA2 支持 [[JVP]]（Jacobian-vector product），用于连续时间 sCM 的 TF 掩码切线计算；稀疏 TF 掩码表达为 admissible query-key 区间，无需物化密集掩码

## 相关概念

- [[自注意力]]
- [[Block Sparse Attention]]
- [[Transformer]]
