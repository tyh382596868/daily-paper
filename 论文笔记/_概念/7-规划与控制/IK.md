---
type: concept
aliases: [IK, Inverse Kinematics, 逆运动学]
---

# IK

## 定义
给定末端执行器的目标位姿（位置 + 姿态），求解满足该位姿的机器人关节角度的过程。是机器人操作和运动控制的基础问题。

## 数学形式
$$\theta^* = \arg\min_\theta \| FK(\theta) - x_{target} \|^2 + \mathcal{R}(\theta)$$

其中 $FK$ 为正向运动学映射，$x_{target}$ 为目标末端位姿，$\mathcal{R}$ 为关节限位等约束。

## 核心要点
1. 封闭形式解（Analytical IK）：特定机器人结构有解析解，速度快但需特定几何配置
2. 数值迭代解：雅可比矩阵伪逆法、CCD 等，通用但可能陷入局部解
3. 多解问题：IK 通常有无数解（冗余机器人）或无解（奇异位形）

## 代表工作
- [[Cloak]]: 用 IK 将掩码末端执行器策略适配到新体态
- [[IOI]]: 运动学网络中用 IK/FK 保证生成视频的几何一致性

## 相关概念
- [[Forward Kinematics]] — 正向运动学，IK 的逆问题
- [[机器人操作]] — 主要应用场景
