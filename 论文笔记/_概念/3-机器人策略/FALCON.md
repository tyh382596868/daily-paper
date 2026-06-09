---
type: concept
aliases: [FALCON humanoid, Force-Adaptive Loco-Manipulation]
---

# FALCON

## 定义

FALCON（Force-Adaptive Humanoid Loco-Manipulation）是一个基于双 Agent 强化学习的人形机器人 loco-manipulation 框架，将全身控制分解为独立的下半身 locomotion Agent 和上半身末端执行器跟踪 Agent，并引入隐式力自适应补偿。

## 核心要点

1. **双 Agent 分解**：下半身 Agent 负责在外力扰动下保持稳定步态，上半身 Agent 负责精确末端执行器位置跟踪
2. **隐式力自适应**：通过接触力的隐式建模实现对未知环境接触力的补偿
3. **操作工作空间相对受限**：与 HANDOFF 对比中，FALCON 的双腕可达凸包在侧向和顶端明显受限

## 代表工作

- FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation (arXiv:2505.06776)

## 相关概念

- [[全身控制]]
- [[locomotion]]
- [[HANDOFF]]
- [[SONIC]]
