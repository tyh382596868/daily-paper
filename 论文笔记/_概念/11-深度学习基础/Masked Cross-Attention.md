---
type: concept
aliases: [掩码跨注意力, Masked Cross Attention]
---

# Masked Cross-Attention（掩码跨注意力）

## 定义

在标准跨注意力机制基础上加入掩码（Mask）的变体，允许控制 Query 对哪些 Key/Value 位置进行注意力计算。常用于多模态融合（如当前帧 → 历史记忆帧）时，防止当前 Query 关注到无效或填充的记忆槽。

## 数学形式

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中 $M$ 为掩码矩阵：有效位置为 $0$，屏蔽位置为 $-\infty$（使 softmax 后权重趋近 0）。

## 核心要点

1. **跨模态/跨时序**: Query 来自当前帧，Key/Value 来自记忆库（历史关键帧），实现时序信息融合
2. **动态长度记忆**: 当记忆库未满时，掩码屏蔽空槽，避免对空位置计算注意力
3. **与 Gated Residual 配合**: 跨注意力产生记忆增强特征 $X'$，随后由 [[Gated Residual Fusion]] 控制注入量

## 代表工作

- [[KEMO]]: 以当前视觉 token 为 Query、关键帧记忆 token 为 Key/Value，做 Masked Cross-Attention 实现记忆融合

## 相关概念

- [[Cross-Attention]]
- [[Gated Residual Fusion]]
- [[Causal Self-Attention]]
