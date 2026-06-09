---
type: concept
aliases: [PCMB, 感知认知记忆库, 感知-认知记忆库]
---

# Perceptual-Cognitive Memory Bank（PCMB）

## 定义

MemoryVLA 系列中提出的核心记忆模块：一个固定容量的双流记忆库，分别存储低层感知细节（感知记忆槽）和高层语义摘要（认知记忆槽），通过带时间位置编码的交叉注意力实现历史上下文检索，并在容量满时通过相邻条目合并维持紧凑性。

## 数学形式

**检索**:

$$\tilde{W}_t = \text{CrossAttn}(Q = W_t + \text{PE}(t),\ K = \mathcal{M},\ V = \mathcal{M})$$

**合并策略**（容量满时）:

$$i^* = \arg\max_{i} \cos(M_i, M_{i+1}), \quad M_{i^*} \leftarrow \frac{M_{i^*} + M_{i^*+1}}{2}$$

## 核心要点

1. **双流存储**：感知槽存低层 RGB 细节（来自 DINOv2），认知槽存高层语义（来自 LLaMA）
2. **时间位置编码**：检索时加入时间步编码，使模型区分"多久之前"的历史
3. **固定容量 $K$**：推理时内存开销为 $O(K)$，不随序列长度增长
4. **合并去冗**：相邻最相似条目均值合并，保留关键历史的同时压缩冗余

## 代表工作

- [[MemoryVLA]]（ICLR 2026）: 首次提出 PCMB
- [[MemoryVLA++]]（arXiv 2606.09827）: 在 PCMB 基础上新增 Imagination Module

## 相关概念

- [[Cross-Attention]]
- [[VLA]]
- [[Imagination Module]]
- [[非马尔可夫任务]]
