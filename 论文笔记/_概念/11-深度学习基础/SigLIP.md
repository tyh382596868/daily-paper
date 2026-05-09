---
type: concept
aliases: [SigLIP, Sigmoid Loss for Language Image Pre-Training]
---

# SigLIP

## 定义
Google 提出的视觉语言预训练方法，用 sigmoid loss 替代 CLIP 的 softmax contrastive loss，支持更大 batch 和更高效训练。

## 数学形式
$$\mathcal{L} = -\frac{1}{n^2}\sum_{i,j} \log \sigma(z_{ij} \cdot (2y_{ij}-1))$$
其中 $z_{ij} = t \cdot \langle v_i, l_j \rangle - b$，$y_{ij} \in \{0,1\}$ 为配对标签。

## 核心要点
1. Sigmoid loss 无需 batch 内的全局归一化，可独立处理每对样本
2. 比 CLIP 更适合分布式训练（无需 all-reduce）
3. 是 [[OpenVLA]] 和 Prismatic VLM 系列的视觉 encoder

## 代表工作
- Zhai et al., 2023: SigLIP 原始论文

## 相关概念
- [[DINO]]
- [[CLIP]]
- [[OpenVLA]]
