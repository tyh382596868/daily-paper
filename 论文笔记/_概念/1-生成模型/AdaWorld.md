---
type: concept
aliases: [AdaWorld, 可适应世界模型]
---

# AdaWorld

## 定义
一类把"latent action"作为可迁移接口的世界模型：先从大量无标注视频里以自监督方式学出一套与具体 embodiment 无关的潜在动作表示，再用它把世界模型快速适配到新环境/新机器人。

## 核心要点
1. 关键在于"latent action 从哪来"——AdaWorld 这条线主张从视频帧间的变化里抽取通用 latent action，而不是依赖动作标注。
2. 学到的 latent action 同时被用于 (a) 训练可泛化的动力学预测、(b) 作为下游 policy 学习/规划的动作空间。
3. 与 UniVLA 等工作同属"latent-action-centric world model / policy"范式，区别主要在 latent action 的抽取方式和是否端到端。

## 代表工作
- [[RLA-WM]]: 指出 latent action 可以直接从 [[DINO]] 残差里学（Residual Latent Action），相比 AdaWorld 这条线 latent action 来源更"白盒"。

## 相关概念
- [[World Model]]
- [[DINO-WM]]
- [[Flow Matching]]
