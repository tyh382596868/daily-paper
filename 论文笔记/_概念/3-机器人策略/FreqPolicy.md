---
type: concept
aliases: [FreqPolicy, Frequency-Domain Policy]
---

# FreqPolicy

## 定义
在频域（DCT 空间）表示机器人动作序列并施加频率感知约束的策略学习方法，解决 action chunking 的边界不连续和高频抖动问题。

## 数学形式
$$\mathcal{L} = \sum_k w_k \|\hat{A}_k - A_k\|^2$$
其中 $\hat{A}_k$ 为 DCT 频域动作系数，$w_k$ 为频率权重（低频系数权重更高）。

## 核心要点
1. 用 DCT 把时域动作序列变换到频域，低频对应整体轨迹平滑，高频对应细粒度调整
2. 频率加权损失让模型优先保证轨迹平滑性
3. FAFM（Frequency-Aware Flow Matching）将此思路与 flow matching 结合

## 代表工作
- [[FAFM]]：Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation（2606.20135）

## 相关概念
- [[DCT（离散余弦变换）]]
- [[Flow Matching]]
- [[Diffusion Policy]]
- [[Action Chunking]]
