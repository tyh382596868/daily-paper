---
type: concept
aliases: [CtrlWorld, Control World]
---

# CtrlWorld

## 定义

一种以轨迹为条件的机器人视频生成方法，在 WorldArena 基准的动作/轨迹条件生成对比组中表现最强（P3CScore 74.03，轨迹精度 0.477）。

## 核心要点

1. 使用轨迹（trajectory）作为条件信号，比原始关节角度提供更多几何信息
2. 在 WorldArena Table 4（轨迹/动作条件对比）中轨迹精度（0.477）超越 EA-WM 的无 GT KVAF 版本
3. 但在使用 GT KVAF 条件时，EA-WM 在轨迹精度（0.494）和总分（78.13）上超越 CtrlWorld

## 代表工作

- [[EA-WM]]: 在 Table 4 中与 CtrlWorld 直接对比

## 相关概念

- [[世界模型]]
- [[KVAF]]
- [[视频扩散模型]]
