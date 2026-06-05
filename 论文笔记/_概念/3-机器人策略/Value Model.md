---
type: concept
aliases: [Value Model, 价值模型, V-model]
---

# Value Model

## 定义

Value Model 是给定状态（或状态-动作对）输出标量"好坏"评分的模型，源自强化学习中的价值函数 $V(s)$ 或 $Q(s, a)$。在 [[Test-Time Scaling|测试时扩展]] 流程中常被用于对候选动作的想象未来打分，挑出最优执行。

## 核心要点

1. **标量评分**: $V: \mathcal{S} \to \mathbb{R}$
2. **训练方式**: 监督学习（专家轨迹）、TD 学习、对比学习
3. **TTS 用途**: 对 $K$ 个候选动作的想象未来打分，$\arg\max_k V(o^{(k)}_{t+n})$
4. **效率瓶颈**: 必须比策略 + 世界模型推理更快，否则 TTS 不划算

## 代表工作

- [[WLA]]: 用价值模型为 TTS 选最优动作
- [[CLIP-Score]]: 用 CLIP 作为零成本评分代理
- 经典 RL: V-function、Q-function

## 相关概念

- [[Test-Time Scaling]]
- [[Reinforcement Learning]]
- [[Reward Model]]
