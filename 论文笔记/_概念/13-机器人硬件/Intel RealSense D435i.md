---
type: concept
aliases: [Intel RealSense D435i, RealSense D435i, D435i]
---

# Intel RealSense D435i

## 定义

**Intel RealSense D435i** 是 Intel 推出的紧凑型双目立体深度相机，集成 IMU，是机器人/无人机最常用的深度感知硬件之一。

## 关键参数

- 输出：深度图 + RGB + IMU
- FOV: $87° \times 58°$
- 最高深度帧率：90 Hz（实际机器人应用常用 30 Hz）
- 最小深度：约 0.28 m
- 重量：约 75 g（无外壳）

## 在 [[MAD]] 中的使用

- 唯一感知传感器
- 30 Hz 深度流，降采样到 $18 \times 32$ 输入模型
- 不使用 IMU 与 RGB（仅深度 + 来自飞控的本体感知）

## 关联概念

- [[Quadrotor]]
- [[深度相机]]
- 同类硬件：D455、L515、ZED 2 等
