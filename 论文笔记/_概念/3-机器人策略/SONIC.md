---
type: concept
aliases: [SONIC, Supersizing Motion Tracking, SONIC全身控制器]
---

# SONIC

## 定义

NVIDIA 提出的全身运动跟踪控制器（Supersizing Motion Tracking for Natural Humanoid Whole-Body Control），通过大规模物理运动模仿训练，实现人形机器人对参考动作轨迹的全身跟踪，作为任务通用底层控制器被广泛复用。

## 核心要点

1. 在大规模运动数据上（包含多样全身动作）通过物理模拟训练，具备泛化的全身运动跟踪能力
2. 作为"冻结"底层控制器，上层适配器（如 GRAIL 的物体感知潜变量适配器）通过调制其潜变量 token 实现任务扩展
3. 开源，可与 Isaac Lab 集成，支持 Unitree G1 等人形机器人平台

## 代表工作

- [[GRAIL]]: 在 SONIC 上训练物体感知适配器和场景感知高度图跟踪器，实现 loco-manipulation
- [[SONIC 原论文]]: NVlabs/GEAR-SONIC，NVIDIA DAIR 实验室
- [[HANDOFF]]: 以 SONIC 为速度跟踪基线，HANDOFF 达到相同水平，同时提供更大的操作工作空间

## 相关概念

- [[DeepMimic]]
- [[PPO]]
- [[IsaacLab]]
- [[loco-manipulation]]
