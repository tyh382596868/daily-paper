---
type: concept
aliases: [BPE, Byte Pair Encoding, 字节对编码, Byte-Pair Encoding]
---

# 字节对编码 (BPE)

## 定义

一种迭代式子词（subword）词表构建算法：反复合并最高频的相邻符号对，直到达到目标词表大小；广泛用于 NLP tokenizer（GPT、LLaMA 等），也用于压缩离散符号序列（如量化后的动作系数）。

## 数学形式（算法）

$$
\text{merge}(a, b) = ab \quad \text{if } \text{count}(ab) = \max_{(x,y)} \text{count}(xy)
$$

迭代执行直到词表达到目标大小 $V$。

## 核心要点

1. **无损压缩**: BPE 通过合并高频对减少序列长度，不丢失信息（可逆）。
2. **动态粒度**: 常见模式用单个 Token 表示，罕见模式拆为多个 Token，自适应分配词表容量。
3. **跨模态扩展**: 不仅限于文本，UniVLA 将其用于量化后的 DCT 动作系数，实现动作 Token 化。
4. **与 SentencePiece / WordPiece 对比**: BPE 从字符起步合并，是最广泛使用的子词 tokenization 方法。

## 代表工作

- GPT 系列: 使用 BPE 作为文本 tokenizer
- [[UniVLA-ICLR2026]]: 将 BPE 应用于 DCT 系数序列的动作压缩

## 相关概念

- [[离散余弦变换 (DCT)]]
- [[DCT 动作 Token 化]]
- [[Next-Token Prediction]]
