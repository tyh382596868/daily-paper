---
type: concept
aliases: [HOMIE, Humanoid Loco-Manipulation with Isomorphic Exoskeleton]
---

# HOMIE

## 定义
HOMIE（Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit）是一个通过同构外骨骼驾驶舱对人形机器人进行上身远程操控的系统，上身由外骨骼驾驶，下身采用独立 RL 控制器。

## 核心要点
1. 上下身解耦设计：上身由外骨骼（Exoskeleton Cockpit）实时驾驶，下身由独立 RL 策略控制
2. 同构外骨骼设计使人类操作者的动作可以直接映射到机器人上身
3. 优点：精细上身控制；缺点：需要人工实时驾驶，无法自主/语言驱动
4. 无法端到端地从语言指令驱动整个系统

## 代表工作
- [[HANDOFF]]: HANDOFF 通过 VLM 智能体规划器实现了 HOMIE 无法做到的自主语言驱动全身控制

## 相关概念
- [[全身控制]]
- [[Visuomotor Policy]]
- [[HOVER]]
