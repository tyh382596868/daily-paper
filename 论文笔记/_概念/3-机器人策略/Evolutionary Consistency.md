---
type: concept
aliases: [Evolutionary Consistency, 演化一致性, EC]
---

# Evolutionary Consistency

## 定义

在 [[Flow Matching|流匹配]] / 扩散类动作生成中，约束不同去噪时间步预测的速度场彼此一致，使动作意图在整个生成过程中连贯的约束，属于[[多一致性约束]]中针对轨迹演化变换的一致性。

## 核心要点

1. 同一样本采样两个时间步 $\tau_1, \tau_2$，得到两份干净（无对抗扰动）速度场预测，用 L2 距离约束一致。
2. 目的是避免去噪轨迹"中途变心"，让多步生成稳定指向同一任务意图。
3. [[RoVLA]] 中时间步不用均匀采样，而用偏向大 $\tau$（接近干净动作）的 [[Beta 分布时间步采样]]。
4. EC 损失同时也作为生成 [[对抗扰动]] 的目标函数（[[Observational Consistency|OC]] 沿 $\nabla\mathcal{L}_{\text{EC}}$ 加扰）。

## 数学形式

$$
\mathcal{L}_{\text{EC}} = \big\| \hat{\mathbf{v}}_{\text{clean}}^{\tau_1} - \hat{\mathbf{v}}_{\text{clean}}^{\tau_2} \big\|_2^2
$$

## 代表工作

- [[RoVLA]]: 用 EC 约束 DiT 流匹配动作头不同去噪步的速度场一致。

## 相关概念

- [[多一致性约束]]
- [[Flow Matching]]
- [[Beta 分布时间步采样]]
- [[Observational Consistency]]
- [[DiT]]
