---
type: concept
aliases: [SPOC, Shortest-Path Oracle Clone]
---

# SPOC

## 定义
"Shortest-Path Oracle Clone"，一种通过模仿最短路径专家(oracle)训练得到的端到端室内导航策略：直接从自我中心 RGB（+ 目标描述）输出离散导航动作，在仿真中大规模克隆专家轨迹。

## 核心要点
1. 学习式策略，推理时不依赖在线大模型，但泛化到未见场景/物体有限。
2. 常作为开放词表物体导航的策略基线。

## 代表工作
- [[3D-Belief]]: 在 AI2-THOR 物体导航中作为"学习式策略"基线（SR 31.67%，远低于 3D-Belief 的 59.17%）

## 相关概念
- [[AI2-THOR]]
- [[5-导航与定位]]
