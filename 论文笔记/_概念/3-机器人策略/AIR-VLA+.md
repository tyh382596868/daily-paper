---
type: concept
aliases: [Aerial Robot VLA Plus]
---

# AIR-VLA+

## 定义
北航提出的 UAV 机械臂 VLA，通过级联双动作解码器 + Asymmetric MoE 解耦飞行控制和末端执行器控制。

## 核心要点
1. 解耦低频全局移动（UAV 飞行）和高频精细操作（机械臂）
2. Asymmetric MoE 差异化分配计算资源
3. 仅仿真验证（has_real_world=False）

## 代表工作
- [[AIR-VLA+]]: arXiv 2606.12859

## 相关概念
- [[DAM-VLA]]
- [[VLA]]
