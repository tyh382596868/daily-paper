---
type: concept
aliases: [Quadrotor, 四旋翼, 无人机, Drone, UAV, Quadcopter]
---

# Quadrotor

## 定义

**Quadrotor（四旋翼）** 是由四个对称分布的旋翼提供升力与控制力矩的旋翼飞行器，是研究级无人机最常用的平台。

## 控制接口层级

- **底层**：电机转速 → 推力 + 力矩
- **中层**：姿态 + 集合推力（attitude + collective thrust）—— [[PX4]]、Betaflight 等飞控接受这一层
- **高层**：位置/速度/加速度命令——经典规划器输出这一层；学习策略则常输出期望加速度命令

## 点质点模型（仿真常用）

$$
\dot{\mathbf{p}}=\mathbf{v},\quad \dot{\mathbf{v}}=\mathbf{a}+\mathbf{g}-d\mathbf{v},\quad \dot{\mathbf{a}}=\lambda(\mathbf{u}-\mathbf{a})
$$

省略姿态动力学，把姿态环交给飞控内环；用于 [[MAD]]、[[DiffAero]] 等学习训练。

## 在自主飞行研究中的关键约束

- 强非线性 + 欠驱动
- 高动力学带宽（需要 100 Hz+ 控制）
- 机载算力受限（典型 NUC 级 CPU / Jetson）
- 唯一感知通常为深度相机（如 [[Intel RealSense D435i]]）

## 关联概念

- [[PX4]]
- [[Intel RealSense D435i]]
- [[MAD]] / [[YOPO]]：端到端学习飞行
- [[Sim-to-Real]]
