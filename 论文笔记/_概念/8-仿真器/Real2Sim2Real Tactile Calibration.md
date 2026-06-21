---
type: concept
aliases: [触觉 Real2Sim2Real 校准]
---

# Real2Sim2Real Tactile Calibration

## 定义
先用真实触觉数据校准仿真触觉模型（Real2Sim），在校准仿真中训练策略，再零样本迁移到真实环境（Sim2Real）。

## 数学形式
$$\theta^*_{sim} = \arg\min \mathbb{E}_{x_{real}}[\|T_{sim}(x, \theta_{sim}) - T_{real}(x)\|]$$

## 核心要点
1. Real2Sim：构建与真实触觉传感器对齐的仿真接触模型
2. 在校准仿真中进行大规模 RL 训练
3. Sim2Real：零样本迁移到真实触觉环境

## 代表工作
- [[Blind Dexterous Grasping]]: 采用 Real2Sim2Real 触觉校准流程

## 相关概念
