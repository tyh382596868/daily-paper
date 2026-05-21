---
type: concept
aliases: [Isaac Lab, Isaac Gym, NVIDIA Isaac Lab]
---

# IsaacLab

## 定义

NVIDIA 推出的**基于 GPU 的并行机器人仿真平台**，继承 Isaac Gym 的并行物理后端，提供大规模并行 RL 训练支持，是目前学术界与工业界 sim-to-real 的主流仿真器之一。

## 核心要点

1. **GPU 并行**: 单卡可同时跑数千个环境实例，训练吞吐量远超传统 CPU 仿真
2. **PhysX 后端**: NVIDIA 自研物理引擎，支持刚体、铰接、布料、流体（有限）
3. **任务套件**: 内置 Franka 操作、足式 locomotion、quadrotor、cube manipulation 等基准
4. **资产格式**: USD / URDF，与 [[3DGS]] / Omniverse 生态集成
5. **不可微**: PhysX 默认不可微，需要替代方案（如 [[OrbiSim]]、Warp、Brax）才能做解析梯度优化

## 代表工作

- 大量 sim-to-real 工作的训练平台
- [[OrbiSim]]: 用 IsaacLab Stack 任务做长 horizon 仿真评测

## 相关概念

- [[MuJoCo]]
- [[ManiSkill]]
- [[RoboTwin]]
- [[sim-to-real]]
- [[Embodied AI]]
