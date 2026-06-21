---
type: concept
aliases: [Vision-Language Navigation BERT]
---

# VLNBERT

## 定义
基于 BERT 的视觉-语言导航模型，将语言指令和视觉观测联合编码用于机器人路径跟随。

## 数学形式
$$P(a_t | I, v_{1:T}) = \text{softmax}(W \cdot \text{BERT}([l_{1:N}; v_{t-k:t}]))$$

## 核心要点
1. BERT 双向编码器联合处理语言指令和视觉特征
2. Cross-attention 实现语言-视觉对齐
3. 在 R2R 等 VLN benchmark 上取得 SOTA

## 代表工作
- [[WoMAP]]: 与 VLNBERT 等 VLN 方法作为基线对比

## 相关概念
