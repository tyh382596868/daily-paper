---
type: concept
aliases: [情境检索, In-Context Retrieval, 检索增强情境学习]
---

# In-Context Retrieval（情境检索）

## 定义

情境检索是指在情境学习（In-Context Learning）框架下，通过相似度搜索从记忆库或数据集中动态检索与查询最相关的样本，用作情境提示（in-context prompt），而非使用随机或固定的示例。

## 数学形式

$$
\{i_1, \ldots, i_K\} = \arg\!\operatorname{top-}K_{i \in \mathcal{M}}\; \text{sim}(x_q, x_i)
$$

其中 $x_q$ 为查询，$\mathcal{M}$ 为记忆库，$\text{sim}$ 为相似度度量。

## 核心要点

1. **动态适应**: 针对每个新查询检索不同的情境样本，而非使用固定 prompt
2. **相似度度量**: 常用余弦相似度（基于预训练特征提取器）或 L2 距离
3. **记忆库构建**: 通常来自训练集或演示数据，推理时无需更新
4. **效果上限**: 检索效果受记忆库覆盖度限制，极端分布外查询可能检索不到有效样本

## 代表工作

- [[V2T-ICON]] / [[VICX]]: 检索机器人演示的图像-状态对作为情境提示
- RAG（Retrieval-Augmented Generation）: NLP 领域的代表性检索增强框架

## 相关概念

- [[In-Context Learning]]
- [[In-Context Operator Learning]]
- [[V2T-ICON]]
