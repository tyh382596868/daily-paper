---
type: concept
aliases: [Franka, Franka Research Arm, Franka Panda, Franka Emika, Franka FR3]
---

# Franka 研究臂

## 定义

Franka 研究臂（Franka Research Arm，前身为 Franka Emika Panda、现代型号 Franka Research 3 / FR3）是一款由德国 Franka Robotics 公司生产的 7 自由度协作机械臂，因其**力矩控制接口、ROS 集成、模块化关节、高重复精度（±0.1mm）**而成为机器人学习领域最广泛采用的真机平台。

## 核心规格

| 项目 | 参数 |
|------|------|
| 自由度 | 7-DoF（每关节 1 个旋转 DOF） |
| 工作半径 | 855 mm |
| 末端有效载荷 | 3 kg |
| 关节力矩传感器 | 每关节内置 |
| 重复精度 | ±0.1 mm |
| 控制接口 | libfranka / ROS / Polymetis（1kHz） |
| 末端执行器 | Franka Hand 双指夹爪 / 第三方灵巧手 |

## 在机器人学习中的角色

1. **VLA 真机标配**: [[OpenVLA]]、[[π₀]]、[[Pi05]]、[[World-VLA-Loop]] 等众多 VLA 工作的真机评估平台
2. **数据采集主力**: [[DROID 数据集]]、[[BridgeV2]] 等大规模 manipulation 数据集主要用 Franka 采集
3. **支持力控**: 关节力矩传感器使其能完成接触敏感任务（如插拔、装配）
4. **配套生态**: 通常配合 RealSense D435 / D455 相机、Robotiq 夹爪、SpaceMouse 遥操作设备使用

## 典型实验装置（如 [[World-VLA-Loop]]）

- **机器人**: Franka research arm
- **相机**: RealSense D435（第三人称固定视角）
- **任务**: Place Cup、Push Cube 等桌面 manipulation

## 代表工作

- [[OpenVLA]] / [[π₀]] / [[Pi05]] / [[World-VLA-Loop]]: 真机评估平台
- [[DROID 数据集]]: 主要采集平台
- [[OpenVLA-OFT]]: 微调评估

## 相关概念

- [[末端执行器]]
- [[DROID 数据集]]
- [[VLA]]
- [[sim-to-real]]
