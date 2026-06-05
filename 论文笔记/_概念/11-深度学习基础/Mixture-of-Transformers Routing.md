---
type: concept
aliases: [MoT Routing, MoT 路由]
---

# Mixture-of-Transformers Routing

## 定义

**Mixture-of-Transformers Routing** 是 [[MoT|Mixture-of-Transformers]] 架构中的核心机制：在每个 Transformer 层为不同 token 子序列（按模态或任务划分）路由到独立的 FFN / attention 投影 expert 参数，但共享 attention softmax 计算空间。

## 核心要点

1. **静态路由**: 路由由 token 模态预先决定，不像 [[Mixture of Contrastive Experts|MoE]] 那样依赖动态 gating
2. **参数翻倍但激活不翻倍**: 每个 token 只走一组 expert，FLOPs 不增
3. **共享 attention**: 不同 expert 输出依然通过同一 softmax 互相 grounding
4. **应用场景**: omni-modal 模型避免不同模态梯度互相干扰

## 代表工作

- [[Cosmos3]]: AR / DM 两组 expert，按模态静态路由
- [[MoT]]: 概念提出

## 关联

- [[MoT]]: 基础概念
- [[Two-Tower MoT]]: 实例化
- [[JointSelfAttention]]
