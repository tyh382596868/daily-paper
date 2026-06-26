---
type: concept
aliases: [Sim-to-Real via Adaptive Policy Gradient, Sequential Anyplay Policy Gradient]
---

# SAPG

## 定义
一种面向灵巧手操作的 sim-to-real RL 框架，通过并行仿真 + 自适应策略梯度实现从仿真到真实机器人的零样本迁移，常用于精密装配等 contact-rich 任务。

## 核心要点
1. 并行化 IsaacGym/IsaacLab 仿真，大规模 rollout 加速 RL 探索
2. 自适应策略梯度处理多指手的高维动作空间
3. 可用于 Play2Perfect 这类 dexterous pretraining 研究
4. 重点克服 contact-rich 任务中的 sim-to-real gap

## 相关概念
- [[Diffusion Policy]]
- [[PPO]]
- [[Play2Perfect]]
- [[FurnitureBench]]
