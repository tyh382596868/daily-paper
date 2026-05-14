---
type: concept
aliases: [Passive World Model, 被动世界模型]
---

# passive 世界模型

## 定义

不接收动作条件，仅根据当前观测预测未来的[[世界模型]]，对应"被动观察"环境演化的范式。

## 数学形式

$$
p(o_{t+1:t+k} \mid o_t,\ l) = \int p(o_{t+1:t+k},\ a_{t+1:t+k} \mid o_t,\ l)\, da
$$

即把联合分布在动作维度上边缘化。

## 核心要点

1. 通用视频生成模型本质上是 passive WM
2. 在机器人场景中只能用于规划"想象式"目标，无法做闭环动作评估
3. 与 [[Controllable 世界模型]] 形成对偶 — 二者是同一联合分布的两种边缘化

## 代表工作

- 通用视频扩散模型（[[CogVideoX]]、[[Wan2.2-TI2V]] 等基础形态）
- 早期 video plan 方法（UniPi 训练阶段、Dreamitate）

## 相关概念

- [[世界模型]]
- [[Controllable 世界模型]]
- [[RobotWM-Survey]]
