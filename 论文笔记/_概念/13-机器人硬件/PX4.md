---
type: concept
aliases: [PX4, PX4 Autopilot]
---

# PX4

## 定义

**PX4** 是开源的无人机飞控固件栈，支持四旋翼、固定翼、VTOL 等多种平台。提供姿态环 / 速率环 / 位置环的级联控制，并暴露多个 setpoint 接口（位置、速度、姿态 + 集合推力、机体 rate）。

## 在学习驱动飞行中的角色

- **底层稳定保障**：学习策略通常输出期望加速度或姿态 + 推力 setpoint，由 PX4 内环把它执行到电机。
- **Sim-to-Real 的隐含组件**：很多 [[Sim-to-Real|sim-to-real]] 成功案例其实部分依赖 PX4 内环的鲁棒性来吸收仿真到现实的动力学差异。
- 与 [[MAD]] 中的接口：MAD 策略输出局部系下期望推力加速度 → 转换为姿态 + 集合推力 → 喂给 PX4。

## 关联概念

- [[Quadrotor]]
- [[Sim-to-Real]]
- Betaflight / Ardupilot：同类飞控固件
