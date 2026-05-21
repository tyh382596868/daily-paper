---
type: concept
aliases: [Multi-Query Attention, GQA, Grouped-Query Attention, 多查询注意力, 分组查询注意力]
---

# MQA / GQA

## 定义

Transformer 注意力的内存优化变体：**Multi-Query Attention (MQA)** 让所有 query head 共享同一组 key/value；**Grouped-Query Attention (GQA)** 把 head 分组、组内共享 KV。两者都显著减小 KV cache 显存与带宽消耗。

## 数学形式

标准 MHA：每个 head $i$ 有独立的 $W_K^i, W_V^i$；
MQA：所有 head 共享同一组 $W_K, W_V$；
GQA：head 分成 $g$ 组，组内共享。

KV cache 大小 ∝ head 数量。

## 核心要点

1. 推理阶段 KV cache 是显存瓶颈，MQA/GQA 大幅缓解；
2. 性能损失通常很小，特别是 GQA 几乎无损；
3. [[AR-VLA]] 的 Action Expert 使用 query=2048D / KV=256D（比例 8:1），是典型 GQA 结构；
4. 现代 LLM（LLaMA-2、Mistral、[[Gemma]]）普遍采用。

## 代表工作

- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)

## 相关概念

- [[Transformer]]
- [[KV Cache]]
- [[Hybrid KV Cache]]
- [[自注意力]]
