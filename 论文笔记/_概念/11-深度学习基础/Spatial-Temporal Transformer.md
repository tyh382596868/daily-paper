---
type: concept
aliases: [ST-Transformer, 时空Transformer, Spatial Temporal Transformer Block]
---

# Spatial-Temporal Transformer (ST-Transformer)

## 定义
一种同时建模视频空间维度和时间维度关系的 Transformer 变体，通过分离的时间注意力层和空间注意力层，在视频理解与生成任务中高效捕获时序动态和空间结构。

## 数学形式

$$
h' = \text{TimeAttn}(h) \quad \text{（时间维度，因果）}
$$
$$
h'' = \text{SpaceAttn}(h') \quad \text{（空间维度，双向）}
$$

## 核心要点
1. **时间注意力层**: 通常采用因果自注意力（Causal Self-Attention），保证生成时的时序因果性（不依赖未来帧）
2. **空间注意力层**: 通常采用双向注意力（Bidirectional Attention），对单帧内所有 patch 建立全局上下文
3. **解耦设计**: 时间和空间维度分开处理，减少计算量，且更易于分别优化各维度的建模能力

## 代表工作
- [[RoboScape]]: DCT 中的核心模块，每个分支由堆叠 ST-Transformer 块构成

## 相关概念
- [[因果自注意力]]
- [[双向注意力]]
- [[Dual-Branch Co-Autoregressive Transformer]]
- [[Transformer]]
