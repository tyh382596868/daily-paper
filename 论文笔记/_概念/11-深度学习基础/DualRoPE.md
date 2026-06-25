---
type: concept
aliases: [Dual RoPE, Dual Rotary Position Embedding]
---

# DualRoPE

## 定义
面向视频生成的双流旋转位置编码方案：对参考帧（reference frame）和生成帧（generated frame）分别使用独立的 RoPE，避免两类帧之间的位置耦合干扰，用于 subject-driven 视频生成。

## 核心要点
1. 标准 [[RoPE]] 对所有帧统一编码，导致参考帧"身份信息"和生成帧"时序信息"混淆
2. DualRoPE：参考帧使用固定位置编码，生成帧使用时序位置编码，实现解耦
3. 让 cross-attention 能更精准地从参考帧提取主体特征
4. 在 [[DomainShuttle]] 中用于 open domain subject-driven T2V

## 代表工作
- [[DomainShuttle]]: 使用 DualRoPE 实现参考主体与生成内容的位置解耦

## 相关概念
- [[RoPE]]
- [[3D mRoPE]]
- [[AnimateDiff]]
