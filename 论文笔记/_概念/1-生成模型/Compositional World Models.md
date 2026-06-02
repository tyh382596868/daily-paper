---
type: concept
aliases: [组合式世界模型, Compositional World Model]
---

# Compositional World Models

## 定义

具有「组合泛化」能力的世界模型 — 可以把训练时见过的元素（物体、场景、机器人、动作）以**未见过的组合**方式渲染出有物理一致性的新观测。区别于"端到端文本条件"或"in-place 视觉增广"路线。

## 核心要点

1. 强调对生成过程的**因子化建模**（factor of variation）：物体、场景、运动学独立可控
2. 是 scalable robot data synthesis 的关键 — 把 $N$ 条真实数据组合扩展到 $N^k$ 种新组合
3. 与 [[AdaWorld]]（latent action factorization）形成对照：CWM 更强调显式的物理因子分解，AdaWorld 强调动作 token 的可迁移
4. 区别于纯 in-place 增广（[[ROSIE]] / [[RoboEnvision]]）— 后者只能换皮，不能生成新物理配置

## 代表工作

- [[RoboDream]]: 三路条件（运动学/场景/物体）的组合式世界模型

## 相关概念

- [[Compositional Conditioning]]
- [[Controllable 世界模型]]
- [[World Model]]
- [[AdaWorld]]
- [[Factors of Variation]]
