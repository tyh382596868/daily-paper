---
type: concept
aliases: [Controllable Human-Object Interaction Synthesis]
---

# CHOIS

## 定义
CHOIS 是一类基于扩散模型的 Controllable Human-Object Interaction Synthesis 方法，给定文本/物体几何/接触约束等条件，合成一段同时驱动人体姿态（[[SMPL]]）和物体 6D 位姿的运动序列。常用作 humanoid loco-manipulation 数据扩增的中间表征生成器。

## 核心要点
1. **输入**：文本指令 + 物体几何 / 初始位姿 + 可选接触/路径约束
2. **输出**：一段 SMPL 序列 + 物体 6D 位姿序列，保持接触一致性
3. **典型架构**：transformer 主干 + diffusion 训练目标 + classifier-free guidance 控制接触约束
4. **下游**：合成数据可用于 humanoid 控制策略训练（如 [[GRAIL]] -> [[ResMimic]]）

## 代表工作
- 原论文：Li et al., "CHOIS: Controllable Human-Object Interaction Synthesis", ECCV 2024 / arXiv 2312.03913
- [[GRAIL]] 在其 pipeline 里借鉴/对比 CHOIS

## 相关概念
- [[SMPL]]
- [[HOI]]
- [[GENMO]]
- [[GRAIL]]
