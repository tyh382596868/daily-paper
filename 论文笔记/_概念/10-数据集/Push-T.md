---
type: concept
aliases: [Push-T, PushT]
---

# Push-T

## 定义

Push-T 是 [[Diffusion Policy]] (Chi et al., 2023) 引入的 2D 操作 benchmark：智能体（圆形 agent）在 2D 平面内推一个 T 形物块到指定位姿。任务对接触动力学和长 horizon 推理都有要求，是机器人模仿学习与世界模型的常用评测环境。

## 数学形式

- **状态**：agent (x, y)、T-block (x, y, θ)
- **动作**：连续 2D 速度 / 位置控制
- **奖励 / 成功率**：T 形覆盖目标区域的 IoU 或目标姿态距离

## 核心要点

1. **接触丰富 → 难规划**：T 形与圆形 agent 的非平滑接触动力学是主要挑战
2. **小型但代表性**：尺寸小、仿真快，便于做大规模消融
3. **被广泛采用**：Diffusion Policy、3D-DP、[[LeWM]]、[[DINO-WM]] 均报 Push-T 结果

## 代表工作

- Chi et al., 2023: Diffusion Policy（首次引入 Push-T benchmark）
- [[LeWM]]: 在 Push-T 上一致超越 PLDM 和 DINO-WM

## 相关概念

- [[OGBench]]
- [[世界模型]]
