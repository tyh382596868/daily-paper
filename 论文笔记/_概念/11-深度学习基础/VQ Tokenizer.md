---
type: concept
aliases: [VQ 分词器, 视觉分词器, Visual Tokenizer, Image Tokenizer]
---

# VQ Tokenizer

## 定义

基于向量量化（Vector Quantization）的图像分词器，将连续图像像素空间映射为离散 codebook index 序列，使图像可作为 token 输入语言模型。

## 数学形式

编码过程：

$$
z_q = \arg\min_{e_k \in \mathcal{C}} \| \text{Enc}(x) - e_k \|_2
$$

训练损失（VQ-VAE 目标）：

$$
\mathcal{L} = \| x - \text{Dec}(z_q) \|^2 + \| \text{sg}[z_e] - e \|^2 + \beta \| z_e - \text{sg}[e] \|^2
$$

## 核心要点

1. **码本大小**: 通常 8,192 ～ 32,768 个 codebook 向量（如 Emu3 用 32,768）
2. **压缩比**: 512×512 图像 → ~1024 token（约 256:1 压缩）
3. **无损信息**: 对 codebook 足够大时可接近无损重建
4. **与 LM 统一**: 离散 token 与文本 token 共享词表，支持联合建模

## 代表工作

- [[VQ-VAE]]: 基础架构
- [[VQGAN]]: 加入对抗训练提升感知质量
- [[Emu3]]: Emu3-VisionTokenizer，codebook 32,768，用于 UniVLA
- [[UniVLA]]: 使用 Emu3 VQ Tokenizer 统一视觉-语言-动作建模

## 相关概念

- [[VQ-VAE]]: 底层量化机制
- [[Autoregressive Transformer]]: VQ Tokenizer 的下游用户
- [[FAST Action Tokenizer]]: 动作模态的类似分词器
