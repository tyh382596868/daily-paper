---
type: concept
aliases: [Action from Video Diffusion, AVDC]
---

# AVDC

## 定义

AVDC（Action from Video Diffusion Classifier）是一种层次化机器人操作框架，利用视频扩散模型预测未来视觉动态，再通过逆动力学模型从预测帧中提取机器人动作。

## 核心要点

1. 使用视频扩散模型生成未来观测帧作为高层次视觉规划
2. 通过相邻帧之间的逆动力学恢复具体动作
3. 在 Meta-World 11 个任务上训练，展示了一定的跨任务泛化能力

## 代表工作

- [[AVDC 原论文]]: 提出框架的原始论文
- [[VICX]]: 以 3 个源任务超越 AVDC（11 个任务训练）的后续工作

## 相关概念

- [[视频规划]]
- [[逆动力学模型]]
- [[视频生成模型]]
