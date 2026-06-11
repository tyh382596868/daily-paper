---
type: concept
aliases: [Multi-view Transformer, 多视图变换器]
---

# 多视图 Transformer

## 定义
在多张输入视图的 token 间做（跨视图）注意力的 Transformer，用于在前馈式 3D 重建/新视角合成中聚合多视图信息、保证几何一致性。

## 核心要点
1. 跨视图 attention 让每个视图"看到"其他视图，隐式完成对应关系建模。
2. 常与位姿编码 / [[代价体|Cost Volume]] / epipolar 约束结合。
3. 是 VGGT、DUSt3R 类前馈几何模型的骨架。

## 代表工作
- [[3D-Belief]]: 几何头用多视图 Transformer + 代价体保证跨视图几何一致
- [[VGGT]]: 用多视图 Transformer 前馈预测相机位姿、深度、点图

## 相关概念
- [[Transformer]]
- [[代价体]]
- [[3D Gaussian Splatting]]
