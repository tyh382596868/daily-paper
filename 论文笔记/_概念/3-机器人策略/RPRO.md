---
type: concept
aliases: [Robotic Flow-matching Proximalized Preference Optimization, 近端偏好优化]
---

# RPRO

## 定义

Robotic Flow-matching Proximalized Preference Optimization（机器人 flow-matching 近端偏好优化），一种专为 flow-matching VLA 动作头设计的偏好优化目标，通过对比优化器 + 显式近端正则化器消除普通 [[Flow-DPO]] 的奖励黑客失效模式。

## 数学形式

$$
\mathcal{L}_{\text{RPRO}} = \mathcal{L}_{\text{Flow-DPO}} + \lambda \cdot \mathcal{L}_{\text{proximal}}
$$

其中近端正则化项约束隐式奖励的绝对量级：

$$
\mathcal{L}_{\text{proximal}} = \mathbb{E}_{τ^+} \left[ \left( \log \frac{\pi_\theta(τ^+)}{\pi_{\text{ref}}(τ^+)} \right)^2 \right] + \mathbb{E}_{τ^-} \left[ \left( \log \frac{\pi_\theta(τ^-)}{\pi_{\text{ref}}(τ^-)} \right)^2 \right]
$$

## 核心要点

1. 继承 [[Flow-DPO]] 的对比目标（正轨迹得分 > 负轨迹得分），额外添加近端惩罚
2. 近端正则化器锚定 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 的偏离量，防止优化过冲造成奖励黑客
3. 在 Dobot XTrainer 双臂平台四任务上，RPRO 优于 Flow-DPO 及 DAgger/π₀.6*/TPO 等基线

## 代表工作

- [[FlowPRO]]：提出 RPRO 并验证其在真实机器人后训练中的有效性

## 相关概念

- [[Flow-DPO]]（直接前驱，RPRO 在其基础上添加近端项）
- [[DPO]]（原始 DPO 方法）
- [[VLA 后训练]]（应用场景）
- [[Flow Matching]]（目标动作头架构）
