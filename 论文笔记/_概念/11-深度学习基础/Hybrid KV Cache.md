---
type: concept
aliases: [HKV Cache, 混合键值缓存]
---

# Hybrid KV Cache

## 定义

一种为**异步多频率**设计的 KV 缓存结构：对不同更新频率的 token 流分别维护独立的 KV 缓冲——快流用滚动 FIFO，慢流用单槽可刷新缓冲。

## 数学形式

$$
\mathbf{KV}_{\text{total}} = \mathbf{KV}^X_{\text{fast (FIFO)}} \cup \mathbf{KV}^{VL}_{\text{slow (single-slot refresh)}}
$$

## 核心要点

1. **动作流** $\mathbf{KV}^X$：滚动 FIFO，长度 $H$；每步 push 一个新 KV，旧 KV pop；
2. **VL 流** $\mathbf{KV}^{VL}$：单槽缓冲，VLM 推理时整段替换；
3. 解耦快慢线程，控制频率不被感知线程拖慢；
4. 配合 [[Dynamic Temporal Re-anchoring]] 解决两条流位置编码错配问题。

## 代表工作

- [[AR-VLA]]：动作 30Hz、感知 14Hz 解耦，控制延迟降到 29ms

## 相关概念

- [[KV Cache]]
- [[Dynamic Temporal Re-anchoring]]
- [[FIFO]]
- [[自回归动作专家]]
