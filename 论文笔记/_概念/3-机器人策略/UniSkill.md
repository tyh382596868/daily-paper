---
type: concept
aliases: [UniSkill, Cross-Embodiment Skill Representation]
---

# UniSkill

## 定义

UniSkill 是一种跨 embodiment 技能表示学习方法，从人类视频中提取技能向量作为机器人策略的条件输入，实现跨体模仿学习。

## 核心思路

- 通过对比学习或自监督方法在跨 embodiment 视频中学习共享技能表示
- 技能向量作为 demo-conditioned 策略的条件信号
- 解耦技能语义（what）与执行细节（how）

## 核心要点

1. 显式技能空间中间表示，区别于 Vid2Robot 的端对端映射
2. 技能向量可跨 embodiment 迁移（人→机器人）
3. 技能抽象层可能丢失精细空间定位信息，在精度敏感任务上受限

## 代表工作

- [[SeeTraceAct]]: 作为 demo-conditioned VLA 的跨体技能基线，SeeTraceAct 通过轨迹监督弥补了 UniSkill 的空间定位不足

## 相关概念

- [[Cross-Embodiment Learning]]
- [[Demo-Conditioned VLA]]
- [[Imitation Learning]]
