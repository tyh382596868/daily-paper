---
type: concept
aliases: [HOVER, Versatile Neural Whole-Body Controller]
---

# HOVER

## 定义
HOVER（Versatile Neural Whole-Body Controller for Humanoid Robots）是一个用于人形机器人的多功能神经全身控制器，通过暴露 3 点头手接口（头部 + 双手位置）来实现运动与操作的统一控制。

## 核心要点
1. 接收头部和双手的 3D 位置轨迹作为命令输入
2. 每个关键点在控制器频率下需要密集轨迹流，规划器需实时合成这些参考
3. 支持多种运动模式切换（运动/操作）
4. 在 Unitree H1/G1 等人形平台上验证

## 代表工作
- [[HANDOFF]]: HANDOFF 的主要对比基线，HANDOFF 以更紧凑的 10-D 接口替代了 HOVER 的密集轨迹接口

## 相关概念
- [[全身控制]]
- [[MoE]]
- [[Unitree G1]]
