---
type: concept
aliases: [流不稳定性, flow instability, 去噪器不稳定性]
---

# Flow Instability

## 定义

视觉[[世界模型]]幻觉检测信号之一：测量[[Shortcut Flow Matching|流匹配]]推理的 Euler 积分过程中，去噪器对清洁目标预测的步间波动程度，用于检测动作边缘化幻觉。

## 数学形式

$$
u_f = \frac{1}{S/2} \sum_{s=S/2}^{S-1} \| \hat{x}_1^{(s+1)} - \hat{x}_1^{(s)} \|
$$

运动归一化版本：

$$
u_f^{\text{norm}} = \frac{u_f}{m}
$$

## 核心要点

1. 动作条件良好时，去噪器在积分后半段快速收敛，$u_f$ 低
2. 动作边缘化幻觉时，模型对动作不敏感，去噪持续振荡，$u_f$ 高
3. 仅需前向推理即可计算，无需标注或仿真器

## 代表工作

- [[MMBench2]]：与 $u_r$（往返残差）和 $u_s$（跨种子方差）共同组成三元幻觉检测体系

## 相关概念

- [[Round-Trip Residual]]
- [[Inter-Seed Variance]]
- [[Shortcut Flow Matching]]
- [[世界模型]]
