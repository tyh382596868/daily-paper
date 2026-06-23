---
type: concept
aliases: [Multi-view Kinematic Aggregation and Injection, 多视角运动学聚合注入]
---

# MKAI

## 定义

MKAI（Multi-view Kinematic Aggregation and Injection）是 IOI 提出的核心模块，将三视角正交投影的运动学渲染聚合为统一条件表示，并通过加性注入层次化植入视频扩散模型骨干网络。

## 数学形式

$$
C^{\mathrm{agg}} = \mathrm{MKAI}(\{R_k^v\}_{v \in \{f,s,t\}})
$$

注入方式（加性注入）：

$$
h_\ell \leftarrow h_\ell + \mathrm{KineBlock}_\ell(C^{\mathrm{agg}})
$$

## 核心要点

1. **Kinematic Fusion Embedder**: 冻结 VAE 编码三视图潜在特征，加时间嵌入后经共享两层 MLP 聚合
2. **Alignment Embedder**: 将聚合特征 Tokenize 对齐到扩散模型特征空间
3. **Kinematic Blocks**: 加性注入方式层次化植入扩散骨干，优于 AdaLN 和交叉注意力

## 代表工作

- [[IOI]]: 提出 MKAI，在 RoboTwin 2.0 上实现 SOTA FVD 41.23

## 相关概念

- [[正向运动学]]
- [[正交投影]]
- [[视频扩散模型]]
- [[Flow Matching]]
