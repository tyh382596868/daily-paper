---
type: concept
aliases: [AdaManip 数据集, 自适应铰接物体操作数据集]
---

# AdaManip

## 定义

基于 [[IsaacLab|Isaac Gym]] 的**铰接物体 (articulated object) 操作数据集**，覆盖 9 类铰接物体、210 个实例，专门用于评估机器人对带关节物体（抽屉、门、瓶盖等）的自适应操作能力。

## 核心要点

1. **9 类物体 / 210 实例**: 几何与运动学多样性高
2. **每条 episode ~301 步**: 长 horizon
3. **关节约束 dynamics**: 测试模型对约束运动学的建模能力
4. **常作为铰接操作基准**: 评测 VLA / 世界模型在 articulated body 上的表现

## 代表工作

- AdaManip 原论文（提出数据集）
- [[OrbiSim]]: 用其评测 OrbiSim-Dynamics 对铰接物体的仿真能力

## 相关概念

- [[IsaacLab]]
- [[Physion]]
- [[OXE]]
