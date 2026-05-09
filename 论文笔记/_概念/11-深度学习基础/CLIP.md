---
type: concept
aliases: [CLIP, Contrastive Language-Image Pre-Training]
---

# CLIP

## 定义
OpenAI 提出的视觉语言预训练模型，通过对比学习将图像和文本映射到同一语义空间，实现零样本图像分类和跨模态检索。

## 数学形式
$$\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{\exp(\langle v_i, l_i \rangle / \tau)}{\sum_j \exp(\langle v_i, l_j \rangle / \tau)}$$
其中 $v_i, l_i$ 为归一化的图像/文本嵌入，$\tau$ 为可学习温度参数。

## 核心要点
1. 对比学习：正对（配对图文）相似度高，负对（同 batch 内非配对）相似度低
2. 在 4 亿图文对上训练
3. [[SigLIP]] 是其改进版本

## 代表工作
- Radford et al., 2021: CLIP 原始论文

## 相关概念
- [[SigLIP]]
- [[DINO]]
- [[VLM]]
