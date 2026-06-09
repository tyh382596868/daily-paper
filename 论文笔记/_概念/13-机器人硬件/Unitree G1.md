---
type: concept
aliases: [Unitree G1, G1, Unitree G1 人形机器人]
---

# Unitree G1

## 定义

宇树科技（Unitree）推出的全尺寸人形机器人平台，具备 23 个自由度的全身运动能力，支持双手操作和双足行走，是当前人形机器人研究中广泛使用的实验平台之一。

## 核心要点

1. **规格**: 全身 23 DoF，身高约 127cm，体重约 35kg
2. **感知**: 支持头部 RGB 相机（如 Luxonis OAK-D W）和深度感知
3. **计算**: 机载计算有限，通常通过 WiFi 连接桌面 GPU 服务器（如 RTX 5090）进行推理
4. **应用**: 操作任务、地形穿越、全身运动控制的主流实验平台

## 代表工作

- [[GRAIL]]: 使用 G1 进行物体抓取（84%）和爬楼梯（90%）的 sim-to-real 部署验证
- [[HANDOFF]]: G1（29-DoF）+ Dex1-1 拟人夹爪 + ZED-M 相机 + Nvidia Jetson Thor 的完整机载全栈部署

## 相关概念

- [[loco-manipulation]]
- [[SONIC]]
- [[sim-to-real]]
