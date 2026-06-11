---
type: concept
aliases: [离散化分词, 模态离散化, Discrete Token Representation]
---

# Discrete Tokenization

## 定义

将连续信号（图像、音频、动作等）转换为有限词表中的离散 index 序列的技术，使不同模态可以统一为 token 形式输入语言模型。

## 数学形式

对连续信号 $x \in \mathbb{R}^d$，学习映射 $f: \mathbb{R}^d \to \{1, \ldots, K\}$，使得：

$$
\hat{x} = \text{Dec}(\text{codebook}[f(x)])
$$

重建质量：$\min \| x - \hat{x} \|$

## 核心要点

1. **词表复用**: 离散 token 可直接复用 LLM 词表末尾位置，无需修改模型架构
2. **信息压缩**: 通过量化实现数据压缩，去除冗余信息
3. **跨模态统一**: 使视觉、动作与语言 token 在同一序列中出现，支持联合自回归建模
4. **精度-压缩权衡**: codebook 大小决定重建精度；BPE 等进一步压缩 token 长度

## 代表工作

- [[VQ-VAE]]: 图像离散化经典方法
- [[FAST Action Tokenizer]]: 动作的 DCT+BPE 离散化
- [[OpenVLA]]: 使用均匀 bin 进行简单动作离散化
- [[UniVLA]]: 视觉/语言/动作统一离散 token 建模

## 相关概念

- [[VQ Tokenizer]]: 视觉离散化工具
- [[FAST Action Tokenizer]]: 动作离散化工具
- [[Autoregressive Transformer]]: 离散 token 的建模框架
