---
type: concept
aliases: [AI2-THOR, AI2THOR, THOR, ProcTHOR]
---

# AI2-THOR

## 定义
Allen Institute for AI 开发的交互式 3D 室内仿真环境，提供逼真的家居场景、可交互物体、自我中心 RGB-D 渲染与物理模拟，广泛用于具身 AI（导航、操作、问答）研究。ProcTHOR 是其程序化生成大规模房屋的扩展。

## 核心要点
1. 自我中心 RGB + 已知相机位姿，可导出真值 3D 几何/语义/占据，便于评测。
2. ProcTHOR 可程序化生成海量多房间布局，适合训练泛化策略。
3. 常用于物体导航(ObjectNav)、探索、世界模型评测。

## 代表工作
- [[3D-Belief]]: 在 AI2-THOR/ProcTHOR 上训练与评测，并基于它构建 3D-CORE 基准
- [[SPOC]]: 在 AI2-THOR 类环境中模仿最短路径专家训练导航策略

## 相关概念
- [[8-仿真器]]
- [[5-导航与定位]]
