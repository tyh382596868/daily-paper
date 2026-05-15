---
type: concept
aliases: [Context Parallel Training, Sequence Parallel Training]
---

# Context-Parallel Training

参见 [[上下文并行]]。

## 核心要点速览

- 把长序列沿时间/上下文维度切到多卡
- 每卡本地算 partial state / partial attention，跨卡通信合成
- 对 [[Gated DeltaNet]] 等循环模型可用前缀扫描合成转移矩阵 $\mathbf{C}_p$

## 代表工作

- [[SANA-WM]]: 用上下文并行训练 961 帧长视频

## 相关概念

- [[上下文并行]]
- [[Gated DeltaNet]]
- [[线性注意力]]
