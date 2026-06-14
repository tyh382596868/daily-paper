---
type: concept
aliases: [Multi-Tactile Token Synthesizer, 多触觉令牌合成器]
---

# MTTS（Multi-Tactile Token Synthesizer）

## 定义
MTTS 是 FTP-1 提出的跨传感器触觉编码器，将图像型、阵列型、状态型三类异构触觉传感器的输出映射到统一的 token 表示空间，实现跨传感器策略泛化。

## 数学形式
对第 $k$ 种传感器的输入 $\mathbf{s}_k$，通过传感器特定编码器：
$$\mathbf{t}_k = \text{Encoder}_k(\mathbf{s}_k) \in \mathbb{R}^{d}$$

统一 token 空间中做对齐（对比学习目标）：
$$\mathcal{L}_{\text{align}} = \sum_{i \neq j} \text{InfoNCE}(\mathbf{t}_i, \mathbf{t}_j)$$

## 核心要点
1. 解决触觉传感器的跨硬件异质性——21 种传感器的信号差异极大
2. 每种传感器有独立的模态特定编码器，共享 token 空间
3. 与 VLA backbone 接口通用，不需要为每个传感器重新训练策略
4. 配合 UniVTAC 对齐训练

## 代表工作
- [[FTP-1]]: MTTS 提出论文，跨传感器触觉基础策略

## 相关概念
- [[GelSight]]
- [[InfoNCE]]
- [[ACT (Action Chunking Transformer)]]
