---
type: concept
aliases: [First In First Out, 先进先出队列, 滚动缓冲]
---

# FIFO (First In First Out)

## 定义

经典数据结构：最早进入队列的元素最先被取出。在深度学习推理中常用于实现**滚动上下文窗口**——固定容量 $H$，新元素 push 时最旧元素 pop。

## 核心要点

1. 实现简单：环形缓冲区 (ring buffer) 或双端队列；
2. 在 [[Transformer]] 推理中用于维护有界长度的 [[KV Cache]]；
3. [[AR-VLA]] 的 $\mathbf{KV}^X$（动作流）就是 FIFO，长度 $H=20$；
4. 与 LRU 等其他驱逐策略相比，FIFO 在严格因果建模中更自然——丢的总是最早的。

## 代表工作

- [[AR-VLA]]：动作 KV 流的 FIFO 管理
- StreamingLLM 等长上下文推理工作

## 相关概念

- [[KV Cache]]
- [[Hybrid KV Cache]]
- [[滑动窗口]]
