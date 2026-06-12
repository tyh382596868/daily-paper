---
type: concept
aliases: []
---

# Sparse2Act

## 定义
UCLA/Utah 提出的动作对齐稀疏 3D 表示，通过视觉-动作联合监督训练稀疏点云 encoder，提升跨域操作策略迁移能力。

## 核心要点
1. 问题：现有稀疏 3D encoder（SimpleDP3）的跨任务表示对齐差
2. 动作对齐监督：让点云特征直接预测性更强
3. RoPE 位置编码改进空间感知
4. Meta-World + LIBERO-10 跨域验证

## 代表工作
- [[Sparse2Act]]: arXiv 2606.12759

## 相关概念
- [[SimpleDP3]]
- [[DP3]]
- [[RoPE]]
