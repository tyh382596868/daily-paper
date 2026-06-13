---
type: concept
aliases: [Allegro Hand, 灵巧手, allegro-hand]
---

# Allegro Hand

## 定义

由 WONIK Robotics 开发的 16-DoF 欠驱动灵巧机器人手，广泛用于学术界灵巧操作研究，每根手指 4 个自由度，使用无刷直流电机驱动。

## 核心要点

1. **16 个自由度**: 4 根手指（含拇指），每根手指 4-DoF，可执行多样抓取构型
2. **尺寸偏大**: 约为成人人手的 2 倍，限制了可操作小型工具的能力
3. **位置控制接口**: 通过 CAN 总线通信，提供关节位置控制和力矩控制模式
4. **广泛集成性**: 与 IsaacLab、MuJoCo 等主流仿真平台有良好的 URDF 支持

## 代表工作

- [[Mana]]: 使用 Allegro Hand + xArm7 实现关节型工具灵巧操作，零样本 sim-to-real 迁移

## 相关概念

- [[IsaacLab]]
- [[Sim-to-Real]]
- [[Diffusion Policy]]
