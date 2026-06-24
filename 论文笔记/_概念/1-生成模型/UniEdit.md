---
type: concept
aliases: [UniEdit, Unified Video Editing]
---

# UniEdit

## 定义
统一视频编辑框架，支持文字引导的内容、风格、结构等多类型视频编辑，通过 attention map 操控实现精细的编辑控制，保留视频时序一致性。

## 核心要点
1. 基于预训练扩散模型，无需额外训练
2. 注意力 injection 机制：保留非编辑区域的结构和运动
3. 时序一致性：利用 cross-frame attention 保持视频连贯性

## 代表工作
- [[UniEdit]]: Bai et al., 2024，视频编辑框架
- [[IOI]]: 将 UniEdit 作为交互式世界模型的对比基线

## 相关概念
- [[ControlNet]] — 条件控制机制
- [[视频扩散模型]] — 基础框架
- [[交互式世界模型]] — 应用场景
