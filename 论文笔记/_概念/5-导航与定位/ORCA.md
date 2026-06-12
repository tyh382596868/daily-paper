---
type: concept
aliases: [Optimal Reciprocal Collision Avoidance]
---

# ORCA (Optimal Reciprocal Collision Avoidance)

## 定义
经典社交导航算法，通过最优速度约束（velocity obstacles）实现多智能体无碰撞导航，常用作社交导航 baseline。

## 数学形式
$$V_A^\text{new} \in \text{ORCA}_{A|B}^\tau$$

## 核心要点
1. 基于速度障碍（velocity obstacles）概念
2. 每个 agent 独立计算无碰撞速度集合
3. 不需要显式通信，分布式实现

## 相关概念
- [[SARL]]
- [[KinematicRL]]
