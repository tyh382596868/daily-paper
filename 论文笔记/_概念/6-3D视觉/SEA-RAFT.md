---
type: concept
aliases: [SEA-RAFT, Simple Efficient Accurate RAFT]
---

# SEA-RAFT

## 定义
SEA-RAFT 是 RAFT 系列光流估计网络的一个简化、高效、精确的变体，用于从图像对中端到端预测[[光流]]场。

## 核心要点
1. 继承 RAFT 的迭代式 recurrent 光流更新框架，简化结构并提升速度与精度。
2. 输出稠密的逐像素二维运动向量场，可用于下游运动分析与评估。
3. 在视频生成/世界模型评估中作为现成的光流提取工具。

## 代表工作
- [[PROWL]]: 用 SEA-RAFT 在解码像素帧上计算光流，构造 Action-Follow Score（AFS-EPE）评估指标

## 相关概念
- [[光流]]
- [[Action-Follow Score]]
