---
type: concept
aliases: [StaInf, 状态推断模块, Real-to-Sim 状态推断]
---

# State Inference

## 定义

OrbiSim 的 Real-to-Sim 子模块之一，从**单帧 RGB 图像**推断可见物理状态：机器人末端位姿、物体位置/旋转/尺寸、视觉属性。

## 数学形式

$$
(x_t^{visible}, \bar{x}^{vis}) = f_\phi^{StaInf}(o_t)
$$

实现上复用冻结的 [[OrbiSim-Vision]] encoder 提取 latent，再接轻量回归头。

## 核心要点

1. **单帧输入**: 只用一张 RGB 即可初始化仿真场景
2. **共享视觉表示**: encoder 与 [[OrbiSim-Vision]] 共用，避免重复训练
3. **结构化输出**: 直接给出位置/旋转/尺寸数值，可送入物理仿真器
4. **精度**: 物体位置 ~27 mm，旋转 7.37°，尺寸 5.2%，机器人位置 22 mm

## 代表工作

- [[OrbiSim]]: 提出 StaInf 与 [[Physics Inference|PhyInf]] 组成 Real-to-Sim pipeline

## 相关概念

- [[Physics Inference]]
- [[OrbiSim-Vision]]
- [[逆动力学]]
