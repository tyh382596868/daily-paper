---
type: concept
aliases: [FAST, DCT Action Tokenizer, 频域动作分词器]
---

# FAST Action Tokenizer

## 定义

一种基于离散余弦变换（DCT）的动作分词器，将连续机器人动作序列变换到频域后进行量化和 BPE 编码，生成紧凑的离散 token 序列，用于 LLM 式自回归动作预测。

## 数学形式

$$
\hat{a} = \text{DCT}(a_{1:T}), \quad \hat{a}_{quant} = \text{BPE}(\text{Quantize}(\hat{a}))
$$

其中 $a_{1:T} \in \mathbb{R}^{T \times D}$ 为 $T$ 步 $D$ 维连续动作序列。

## 核心要点

1. **频域压缩**: DCT 将时域动作序列变换为频域系数，低频成分捕捉宏观运动，高频捕捉细节
2. **信息保留**: 相比逐维度均匀 bin（OpenVLA），更好地保留了动作的时序相关性
3. **BPE 进一步压缩**: 对量化后的频域系数做 Byte-Pair Encoding，减少 token 数量
4. **数据集特化**: 对不同任务的动作分布独立 fit（如 `fast_calvin_norm_a10_s50`）

## 代表工作

- [[FAST]]: 原始论文（Hejna et al., 2025，arXiv:2501.09747）
- [[UniVLA]]: 将 FAST 集成到统一 VLA 框架的代表工作
- [[pi0-FAST]]: π₀ 使用 FAST 动作分词器的版本

## 相关概念

- [[DCT（离散余弦变换）]]: 核心变换算子
- [[Action Chunking]]: FAST 处理的动作块粒度（通常 T=10）
- [[Discrete Tokenization]]: 上层概念
- [[OpenVLA]]: 使用简单均匀 bin 的对比方法
