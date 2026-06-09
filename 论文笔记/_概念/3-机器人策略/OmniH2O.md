---
type: concept
aliases: [OmniH2O, Universal Human-to-Humanoid Teleoperation]
---

# OmniH2O

## 定义
OmniH2O（Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning）是一个通用灵巧的人类到人形机器人全身远程操作与学习系统，通过追踪人体头部和双手 3D 位置实现全身运动迁移。

## 核心要点
1. 追踪人体头部（head）和双手（hands）共 3 个关键点的位置（3 点接口）
2. 支持全身运动迁移，包括行走、蹲取、抓取等复杂技能
3. 在真实人形机器人上验证了远程操作和从人类示范学习的可行性
4. 密集轨迹驱动：需要在控制器频率下提供连续轨迹，对规划器要求高

## 代表工作
- [[HANDOFF]]: HANDOFF 将头部追踪替换为骨盆速度，简化了接口同时保留了表达能力

## 相关概念
- [[HOVER]]
- [[全身控制]]
- [[模仿学习]]
