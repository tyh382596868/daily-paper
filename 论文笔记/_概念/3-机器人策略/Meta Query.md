---
type: concept
aliases: [Meta Query, Meta-Query, 元查询]
---

# Meta Query

## 定义

Meta Query 是一组**可学习的 query token**，作为 backbone 与下游 expert head 之间的桥梁。它通过因果/双向注意力聚合 backbone 输出的上下文表征，压缩到固定长度，再供 World / Action / Language Expert 共享解码。

## 核心要点

1. **固定数量 query**: $Q = \{q_1, \dots, q_m\}$，长度与上下文长度无关
2. **共享聚合**: 多个下游 head 共用同一份 meta-query 表征，等价于隐式参数共享
3. **解耦上下文长度**: backbone 上下文可变，meta-query 输出固定，方便 head 设计
4. **类似 Perceiver Resampler**: 思想类似 [[Perceiver IO]]、[[Flamingo]] 的 resampler

## 公式

$$
h_t = \operatorname{CausalAttn}\big(Q, \operatorname{BackboneOut}(\cdot)\big)
$$

## 代表工作

- [[WLA]]: 用 meta-query 把 backbone 输出聚合给 World / Action Expert 共用
- [[Q-Former]]: BLIP-2 的视觉 resampler 思想前身
- [[Perceiver IO]]: 通用 latent query 架构

## 相关概念

- [[Cross Attention]]
- [[Action Chunking]]
