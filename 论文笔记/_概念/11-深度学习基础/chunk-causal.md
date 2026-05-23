---
type: concept
aliases: [Chunk Causal, 块因果, 块级因果]
---

# chunk-causal

## 定义
介于双向与逐 token 自回归之间的注意力掩码模式：将序列按固定大小的"块"切分，块内全部可见，块间仅允许后块看前块。常用于在线视频滚动生成。

## 核心要点

1. **块内双向 / 块间因果**: 既保留小窗口高质量，又支持在线滚动
2. **降低延迟**: 解码新块时只需重算当前块的 attention，旧块 cache 复用
3. **配合 attention sink**: 保留少量全局 token 抗长程退化
4. **典型块长**: 视频任务常用 8-32 帧，60s @ 16fps 视频 ~30-120 块
5. **精度代价**: 与全双向比有 RotErr 增加（[[SANA-WM]] Hard 集 3.17° → 10.02°）

## 代表工作
- [[SANA-WM]]: Stage 4 chunk-causal 微调 + self-forcing 蒸馏
- StreamingLLM、世界模型在线滚动等

> 注: [[EvoScene-VLA]] 中的 "chunk-causal 缺陷" 是策略层面的另一种含义，指动作块内部条件于起始观测、忽略 chunk 中段几何变化的问题。

## 相关概念
- [[softmax 注意力]]
- [[Frame-wise Gated DeltaNet]]
- [[视频扩散模型]]
