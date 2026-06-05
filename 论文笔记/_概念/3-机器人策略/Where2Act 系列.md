---
type: concept
aliases: [Where2Act 系列, Where2Act Family]
---

# Where2Act 系列

## 定义

一类经典 2D affordance map 预测方法的统称，核心思想是给定图像 + 任务 → 输出可交互区域热力图，代表了"先看哪里再动"的传统可供性范式。

## 核心要点

1. **2D 热力图**: 输出像素级 affordance map，标识可推、拉、抓的位置。
2. **指令感知**: 同一物体在不同任务下可供性不同。
3. **局限**: 仅 2D，缺乏 3D 几何，难以直接驱动 6-DoF/10-DoF 控制。
4. **后续发展**: AffordanceVLA 在其基础上加入 Which（对象级）+ How（3D 形状/位姿），形成结构化可供性预测。

## 代表工作

- Where2Act 原文（Mo et al., ICCV 2021）
- [[AffordanceVLA]] 中的 Where2Act 模块（继承名字，集成到 MoT 架构）

## 相关概念

- [[Affordance]]
- [[Where2Act Loss]]
- [[Affordance Forecasting]]
