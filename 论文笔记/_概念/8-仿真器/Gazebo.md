---
type: concept
aliases: [Gazebo, gazebo]
---

# Gazebo

## 定义

**Gazebo** 是开源的高保真机器人仿真器，提供物理引擎、传感器仿真和 ROS 集成。在无人机/机械臂/移动机器人研究中是事实标准评测平台之一。

## 与训练仿真器的区别

- **训练仿真器（如 [[DiffAero]]、[[IsaacLab]]、[[MuJoCo]]）**：追求 GPU 并行、可微、高吞吐，用来 RL 训练。
- **评测仿真器（Gazebo）**：追求物理保真、与真机控制栈接口一致，用来在和真机相同的飞控/ROS 链路下验证策略——是 [[Sim-to-Real|sim-to-real]] 的"中间台阶"。

## 在 [[MAD]] 中的角色

- 与 [[PX4]] 联合做评测：圆柱森林（多稀疏度）、室内走廊、动态障碍场景。
- 评测时使用的控制接口和实机一致（姿态 + 集合推力 → PX4），从而验证策略不依赖训练仿真器的特性。

## 关联概念

- [[Physics Simulator]]
- [[PX4]]
- [[DiffAero]] / [[MuJoCo]] / [[IsaacLab]]
