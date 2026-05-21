---
type: concept
aliases: [UMT5, UMT5-XXL, Unified Multilingual T5]
---

# UMT5

## 定义
UMT5（Unified Multilingual T5）是 T5 系列的多语言文本编码器-解码器模型，常以其编码器部分作为冻结的文本特征提取器，为生成模型提供文本条件嵌入。

## 核心要点
1. 基于 T5 的 encoder-decoder Transformer 架构，在多语言语料上预训练。
2. 在文生视频/文生图模型中，通常只取其 encoder 部分（如 UMT5-XXL）并冻结，将文本 prompt 编码为条件嵌入。
3. 提供的文本嵌入通过 cross-attention 注入扩散主干。

## 代表工作
- [[Wan2.1]]: 用冻结的 UMT5-XXL 编码器处理文本/动作条件
- [[PROWL]]: 复用 Wan2.1 的 UMT5-XXL 编码器，把序列化的动作 token 编码为动作条件；Phase 2 微调其 action-text adapter

## 相关概念
- [[Transformer]]
- [[CLIP]]
- [[Wan2.1]]
