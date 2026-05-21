---
type: concept
aliases: [因果混淆悖论, 因果混淆]
---

# Causal Confusion Paradox

## 定义

模仿学习中的反直觉现象：训练时验证误差越低，部署时成功率反而越低。原因是模型走捷径学到"复制历史动作"而非"根据观测决策"，离线指标无法暴露这种 shortcut。

## 数学形式

定性：

$$
\arg\min_\theta \mathcal{L}_{\text{val}}(\theta) \neq \arg\max_\theta \text{Success}_{\text{rollout}}(\theta)
$$

## 核心要点

1. 历史动作与未来动作高度相关，模型偏好用 $a_{t-1}$ 直接预测 $a_t$；
2. 部署时一旦初始状态偏离，错误会沿着复制链累积；
3. 缓解手段：[[Stochastic History Masking]]、DAgger、控制历史长度、加 noise；
4. [[AR-VLA]] 在 mask=0 时验证误差 2.7（最低），仿真成功率 0%，是典型例证。

## 代表工作

- de Haan et al., "Causal Confusion in Imitation Learning" (NeurIPS 2019)
- [[AR-VLA]]：在跨时间 AR 设置下重新发现并用 mask 缓解

## 相关概念

- [[Stochastic History Masking]]
- [[模仿学习]]
- [[Markovian Amnesia]]
