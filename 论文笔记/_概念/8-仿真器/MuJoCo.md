---
type: concept
aliases: [MuJoCo, Multi-Joint dynamics with Contact]
---

# MuJoCo

## 定义
DeepMind 开发的高效物理仿真引擎，专为机器人控制和强化学习研究设计，以快速、精确的接触动力学著称。

## 数学形式
$$M(q)\ddot{q} + c(q,\dot{q}) = \tau + J^T f$$
其中 $M$ 为惯量矩阵，$c$ 为科里奥利力项，$\tau$ 为广义力，$J^T f$ 为接触力。

## 核心要点
1. 支持软接触模型，避免刚体碰撞的数值不稳定
2. DeepMind Control Suite（DMC）基于 MuJoCo
3. 2021 年被 DeepMind 收购后对研究者免费开放

## 代表工作
- Todorov et al., 2012: MuJoCo 原始论文

## 相关概念
- [[RoboTwin]]
- [[SimplerEnv]]
- [[ManiSkill]]
