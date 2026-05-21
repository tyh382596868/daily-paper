---
type: concept
aliases: [Mamba, 选择性状态空间模型, Selective State Space Model]
---

# Mamba

## 定义
Mamba 是一种基于选择性[[状态空间模型]]（SSM）的序列建模架构——用输入相关的状态空间参数实现可随上下文动态选择信息的线性复杂度序列建模。

## 核心要点
1. 相比 Transformer 的二次注意力，Mamba 在序列长度上为线性复杂度，可高效扩展到长序列。
2. 把历史压缩进固定大小的隐状态，省算力但会损失细节保真度。
3. 在世界模型中常作为「高效但压缩历史」的基线，与全注意力 Transformer 形成效率-保真度的对照。

## 代表工作
- [[CoME]]: 将 Chunked SSM (Mamba) 作为长上下文世界模型的效率基线

## 相关概念
- [[状态空间模型]]
- [[线性注意力]]
- [[Transformer]]
