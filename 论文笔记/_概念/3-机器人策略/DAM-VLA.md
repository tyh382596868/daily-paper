---
type: concept
aliases: [Decoupled Asynchronous Multimodal VLA]
---

# DAM-VLA (Decoupled Asynchronous Multimodal VLA)

## 定义
KIT/NVIDIA 提出的异步多模态 VLA，解耦高频本体感觉流和低频视觉-语言流，解决 VLA 多模态同步时钟问题。

## 核心要点
1. 问题：VLA 预训练假设所有模态同步，但物理交互需要不同频率
2. SPARC 处理高频本体感觉，GCA（Gated Cross-Attention）处理低频视觉-语言
3. 两路异步融合后输出动作
4. KIT 真实机器人验证

## 代表工作
- [[DAM-VLA]]: arXiv 2606.12105

## 相关概念
- [[VLA]]
- [[FASTER]]
