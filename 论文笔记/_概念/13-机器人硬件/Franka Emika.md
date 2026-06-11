---
type: concept
aliases: [Franka, Panda, Franka Panda, Franka Research 3]
---

# Franka Emika

## 定义

Franka Emika（现 Franka Robotics）出品的 7 自由度串联机械臂，因其力矩传感能力、研究友好的 API（libfranka）和适中的价格，成为学术界最流行的机器人操控实验平台之一。

## 数学形式

**关节空间**：$q \in \mathbb{R}^7$（关节角度）

**末端执行器（EEF）位姿**：$T_{EEF} \in SE(3)$（通过正运动学求得）

## 核心要点

1. **7 DoF**: 7 个旋转关节，冗余自由度提高灵活性
2. **并联夹爪**: 标配 Franka Hand（2 指平行夹爪，最大开口 8cm，抓力可控）
3. **力矩传感**: 每个关节集成力矩传感器，支持力控和接触检测
4. **接口**: libfranka（C++）/ franka_ros（ROS）/ franka-py（Python）

## 代表工作

- [[LAD]]: 将 Franka 夹爪作为基准本体，与灵巧手协同训练跨具身操控策略
- [[Diffusion Policy]]: 大量实验在 Franka 上完成

## 相关概念

- [[Faive Hand]]
- [[mimic Hand]]
- [[Cross-Embodiment Learning]]
