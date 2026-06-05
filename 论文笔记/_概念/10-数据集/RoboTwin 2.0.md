---
type: concept
aliases: [RoboTwin 2.0, RoboTwin2.0, RoboTwin-2.0]
---

# RoboTwin 2.0

## 定义

RoboTwin 2.0 是一个大规模双臂机器人仿真基准，包含 50 个 bimanual 任务，提供 2,500 条 clean 轨迹和 25,000 条随机化轨迹用于训练和评估 VLA / WLA 策略。

## 核心要点

1. **50 个任务**: 涵盖 pick-place、stack、insertion、long-horizon 等
2. **两档难度**:
   - **Clean**: 标准场景，无视觉/物理扰动
   - **Randomized**: 物体位置、纹理、光照随机化
3. **大规模**: 2.5k 干净 + 25k 随机化轨迹
4. **双臂**: 强调 [[Bimanual Manipulation|双臂协作]]

## 代表工作

- [[WLA]]: 92.94% Clean / 90.02% Randomized，刷新 SOTA
- [[GR00T N1.7]]: 85.3% / 80.2%
- [[π0]]: 78.1% / 71.4%

## 相关概念

- [[LIBERO]]
- [[Bimanual Manipulation]]
- [[Sim-to-Real]]
