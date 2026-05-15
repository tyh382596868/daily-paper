---
type: concept
aliases: [Context Parallel, CP, 上下文并行]
---

# Context-Parallel 训练

## 定义
把单条长序列切给多张 GPU、跨卡共享必要状态以维持数学等价的并行训练范式。

## 数学形式

对于带循环状态 $S_t$ 的序列模型（如 [[Frame-wise Gated DeltaNet|GDN]]）：

$$
S_{\text{end}}^{(p)} = S_{\text{start}}^{(p)} C_p + H_p, \quad C_p = \prod_{t \in I_p} M_t
$$

其中 $I_p$ 是第 $p$ 卡持有的帧索引区间，$C_p$ 为本地转移合成，$H_p$ 为本地输入累积。

## 核心要点

1. **目标**: 解决长序列（如 60s 视频、961 帧）单卡显存不足问题
2. **状态等价**: 通过独占前缀合成精确恢复每片初始状态，与单卡训练数学等价
3. **通信代价**: 仅 all-gather $O(P)$ 个 $D \times D$ 矩阵，远小于 all-gather 全部激活
4. **配合 halo 交换**: 共享 $K-1$ 帧边界，支持时间卷积
5. **与 GDN 配合最佳**: 因 GDN 状态紧凑（$D \times D$），适合 CP；softmax KV cache 仍需特殊处理

## 代表工作
- [[SANA-WM]]: 用 CP size=2 跨卡训练 961 帧
- 早期工作: Megatron-LM sequence parallelism、Ring Attention

## 相关概念
- [[Frame-wise Gated DeltaNet]]
- [[线性注意力]]
- [[扩散变换器]]
