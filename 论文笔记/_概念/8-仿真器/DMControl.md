---
type: concept
aliases: [DeepMind Control Suite, dm_control, DMC]
---

# DMControl

## 定义

DeepMind Control Suite（DMControl / dm_control）是 DeepMind 基于 [[MuJoCo]] 开发的一组连续控制 benchmark 环境，覆盖 humanoid、walker、cheetah、cartpole、reacher 等经典 3D 控制任务，是 model-based RL 和世界模型研究的事实标准。

## 核心要点

1. **统一接口**：所有任务共享 observation/action/reward 三元组，便于跨任务横评
2. **物理基底**：[[MuJoCo]] 引擎，关节、约束、接触都可精确仿真
3. **任务覆盖**：cheetah-run、walker-walk、humanoid-stand、quadruped-run 等
4. **WM 研究常用**：[[DreamerV3]]、[[TD-MPC2]] 都把 DMControl 作为主战场

## 在 SWM 中的位置

[[StableWM]] 把 DMControl 包装进 `swm.World` 接口，提供 3D locomotion 任务作为 [[Push-T]] / [[TwoRoom]] 之外的复杂度补充。每个 DMControl 环境暴露 6-7 个 [[Factors of Variation]]（关节密度、地面摩擦、lighting 等）。

## 代表工作

- [[StableWM]]: 把 DMControl 接入统一的 World API
- [[DreamerV3]]: 在 DMControl 上的 model-based RL 标杆
- [[TD-MPC2]]: 用 latent dynamics + MPC 在 DMControl 上取得 SOTA

## 相关概念

- [[MuJoCo]]
- [[OGBench]]
- [[Gymnasium]]
