---
type: concept
aliases: [Chunk-Wise Autoregression, 块级自回归, Chunk-wise AR]
---

# Chunk-Wise Autoregression

## 定义

Chunk-Wise Autoregression 是把一个长序列**切成固定长度的块（chunk）**，每个 chunk 内并行预测，chunk 之间做 [[自回归]] 的训练 / 推理范式。相比帧级 [[自回归]]，它解决"相邻 token 差异过小导致 trivial 外推"的问题；相比一次性预测所有未来，它仍保留时间因果。

## 数学形式

把序列 $x_{1:T}$ 切成 $M$ 个 chunk $c_1, c_2, \ldots, c_M$，每个 $c_m$ 含 $K$ 个 token。模型分布：

$$
p(x_{1:T}) = \prod_{m=1}^{M} p(c_m \mid c_{<m})
$$

其中 $p(c_m \mid c_{<m})$ 内部 $K$ 个 token **并行**生成，相邻 chunk 之间保持自回归。

## 核心要点

1. **粒度 trade-off**: chunk 长度 $K$ 平衡"短期一致" vs "长期因果"
2. **跨 chunk stride $s$**: 可以让 chunk 间不相邻，扩大有效视野（如 X-Foresight 的 CLEF 把 stride 从 1 s 拉到 3 s）
3. **训练用 [[Teacher Forcing]]**: 推理时 chunk-by-chunk 滚动
4. **配合 [[Block Sparse Attention|块稀疏 attention]]**: chunk 内双向 attention + chunk 间块稀疏几乎是标配
5. **优于帧级 AR**: 单帧预测信号弱、易走 shortcut；chunk 级强迫模型预测"非平凡未来"

## 代表工作

- [[X-Foresight]]: chunk-wise + CLEF + TIS 把视野推到 21 秒
- 视频生成里的 "future chunk prediction" 范式
- [[GR-2]] / [[CoT-VLA]] 等 chunk-based VLA

## 相关概念

- [[自回归]]
- [[Block Sparse Attention]]
- [[Teacher Forcing]]
- [[Curriculum Learning with Extended Foresight]]
