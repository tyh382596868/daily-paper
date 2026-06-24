---
type: concept
aliases: [Jetson Orin NX, Orin NX, NVIDIA Orin]
---

# NVIDIA Jetson Orin NX

## 定义

NVIDIA Jetson Orin NX 是面向边缘 AI 的嵌入式计算模块，集成 NVIDIA Ampere GPU 核心、ARM Cortex-A78AE CPU 和 NVDLA 深度学习加速器，适用于机器人、无人机等低功耗高算力场景。

## 数学形式

N/A（硬件平台）

## 核心要点

1. GPU：最高 1024 CUDA 核心 + 32 Tensor Core，支持 INT8/FP16 推理
2. 支持 NVIDIA TensorRT 部署，可大幅加速模型推理
3. 功耗低（10–25 W），适合机载嵌入式部署
4. 支持 ROS 2、PX4 等机器人软件栈

## 代表工作

- [[SkyJEPA]]: ~99K 参数的 SkyJEPA 模型通过 TensorRT 部署在 Jetson Orin NX 上，实现 <10 ms 推理延迟，满足 100 Hz 四旋翼控制需求

## 相关概念

- [[PX4]]
- [[World Model]]
