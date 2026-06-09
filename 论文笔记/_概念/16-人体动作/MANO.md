---
type: concept
aliases: [MANO, Hand Model, 手部参数化模型]
---

# MANO (Modeling and Articulation of Hands)

## 定义

手部参数化统计模型，类比 SMPL 对人体的参数化方式，用形状参数 $\beta$ 和姿态参数 $\theta_{hand}$ 描述手部网格，共 778 个顶点、1538 个面片，支持微分可求导。

## 数学形式

$$
M_{hand} = \text{MANO}(\theta_{hand}, \beta_{hand}) \in \mathbb{R}^{778 \times 3}
$$

## 核心要点

1. 与 SMPL-X 兼容：SMPL-X 整合了 SMPL body + FLAME face + MANO hands
2. 参数化表示便于优化（梯度可反传到手部姿态参数）
3. 51 个手部姿态参数（16 个关节 × 3 DoF + 全局旋转）
4. 常用于机器人操作领域的手部动作建模

## 代表工作

- [[GRAIL]]: GEM-SMPL 中通过 WiLoR 估计 MANO 手部参数，用于 HOI 优化器的接触损失约束
- SMPL-X 原论文（Pavlakos et al., 2019）

## 相关概念

- [[SMPL]]
- [[WiLoR]]
- [[HOI]]
