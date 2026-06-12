---
type: concept
aliases: [Tactile-Conditioned RL for VLA]
---

# TacCoRL (Tactile Feedback into VLA via Simulation)

## 定义
UCSD/UESTC 提出的触觉 VLA 训练方法：MimicGen 生成仿真触觉数据 + Co-Training + PPO post-training，实现触觉 sim-to-real 迁移。

## 核心要点
1. MimicGen 生成触觉仿真数据
2. Co-Training：视觉-触觉-动作联合训练
3. SysID + SPSA 校准触觉传感器物理参数
4. 真实机器人验证（has_real_world=True）

## 代表工作
- [[TacCoRL]]: arXiv 2606.11743

## 相关概念
- [[MimicGen]]
- [[Co-Training]]
- [[PPO]]
