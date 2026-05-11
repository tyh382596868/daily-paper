---
type: concept
aliases: [Violation-of-Expectation, VoE, 违反预期]
---

# Violation-of-Expectation

## 定义

Violation-of-Expectation（VoE）原是发展心理学评估婴儿物理直觉的实验范式：当观察到违反物理常识的事件时（物体瞬移、穿墙等），婴儿凝视时间会显著延长。机器学习借用这一框架来**评估世界模型的物理理解**：模型对物理不可能事件应表现出更高的"surprise"。

## 数学形式

给定模型对下一时刻的预测 $\hat z_{t+1}$ 与真实下一观测 $z_{t+1}$，定义 surprise：

$$
\text{surprise}_t = \big\|\hat z_{t+1} - z_{t+1}\big\|_2^2
$$

通过比较 unperturbed / 视觉扰动 / 物理扰动三类轨迹的 surprise 分布是否显著不同（如 paired t-test），判断模型是否"意识到"物理违反。

## 核心要点

1. **物理 vs 视觉**: 一个好的世界模型应对物理违反（瞬移、穿墙）比对视觉扰动（颜色变化）更敏感
2. **评估手段而非训练目标**: VoE 不参与训练，只用于事后评估
3. **可与 latent probing 互补**: probing 测"潜空间是否有物理量"，VoE 测"模型能否区分物理可能 / 不可能"
4. **在世界模型中**: [[LeWM]] 用 VoE 在 TwoRoom、PushT、OGBench-Cube 上验证，物理扰动 surprise 显著高于视觉扰动 ($p<0.01$)

## 代表工作

- Riochet et al., IntPhys (2018): 把 VoE 引入物理直觉的机器评估
- [[LeWM]]: 用 VoE 验证 latent 世界模型的物理敏感性

## 相关概念

- [[World Model]]
- [[JEPA]]
