---
type: concept
aliases: [FLOWER policy, Flow-based VLA policy]
---

# FLOWER

## 定义

FLOWER 是一种基于 [[Flow Matching]] 的 [[VLA]] 动作头：把动作 chunk 看作目标分布的样本，用流匹配从高斯噪声逐步演化到目标动作序列，是 [[CALVIN]] / [[LIBERO]] 上的强基线之一。

## 核心要点

1. **生成式动作头**：把动作生成视作条件分布采样，与 [[Diffusion Policy]] 同属生成式路线；
2. **少步采样**：相比扩散模型，Flow Matching 通常步数更少；
3. **块级输出**：仍按 chunk 一次性输出离散 waypoints；
4. **是 [[NIAF]] 的主要对比对象**：NIAF 把"流匹配采样"换成"连续函数 query"。

## 代表工作

- **FLOWER**：CALVIN ABC→D 4.39 / LIBERO 平均 97.0%
- [[NIAF]]：在同样设置下用连续函数表示超过 FLOWER

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[BEAST]]
- [[Pi05]]
- [[Neural Implicit Action Field]]
