---
type: concept
aliases: [Rotary Position Embedding, 旋转位置编码]
---

# RoPE

## 定义
一种相对位置编码方法，通过旋转矩阵将位置信息编码到 Query 和 Key 向量中，使 attention score 天然包含相对位置信息。

## 数学形式
$$q_m = R_{\Theta,m}^d q, \quad k_n = R_{\Theta,n}^d k$$
$$\text{Attention}(q_m, k_n) = q_m^T k_n = q^T R_{\Theta,m-n}^d k$$

## 核心要点
1. 相对位置感知：attention score 只依赖相对位置 $m-n$
2. 支持外推到训练时未见的序列长度
3. 是 Llama、Qwen 等现代 LLM 的标配位置编码

## 相关概念
- [[LoRA]]
- [[DiT]]
- [[VLA]]
