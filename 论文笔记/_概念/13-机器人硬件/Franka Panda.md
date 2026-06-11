---
type: concept
aliases: [Franka Emika Panda, Panda Robot Arm, FR3]
---

# Franka Panda

## 定义

Franka Emika 出品的 7-DoF 串联机械臂，配备关节力矩传感器和柔顺控制接口（FCI），是学术机器人操作研究中最广泛使用的平台之一。

## 核心要点

1. **7 自由度**: 冗余关节配置允许灵活避障，末端执行器可到达任意姿态
2. **力矩传感器**: 每个关节内置扭矩传感器，支持阻抗控制和接触力感知
3. **FCI 接口**: 提供 1kHz 实时控制接口，支持 libfranka C++ 库和 franka-ros
4. **标准配置**: 通常配备 Franka Hand 平行夹爪（两指），抓取宽度可调
5. **学术广泛性**: 被 RoboAgent、RoboCasa、BridgeData 等主流数据集广泛采用

## 代表工作

- [[SeeTraceAct]]: 使用 Franka Panda 进行真实世界 demo-conditioned 操作实验
- [[TraceVLA]]: 真实机器人评测平台
- [[DP3]]: Diffusion Policy 3D 真实实验平台

## 相关概念

- [[VLA]]
- [[Action Chunking]]
- [[RoboCasa]]
