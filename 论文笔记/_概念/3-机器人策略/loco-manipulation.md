---
type: concept
aliases: [loco-manipulation, 移动操作, 运动操作]
---

# Loco-Manipulation

## 定义

将**移动（locomotion）**与**操作（manipulation）**统一的机器人任务范畴，要求机器人在移动过程中同时与物体进行交互（抓取、搬运、放置等），典型平台为人形机器人或移动操作臂。

## 核心要点

1. 需要全身协调：移动基座、躯干、手臂需要协同控制
2. 挑战来源：接触力控制、运动中的稳定性、视觉感知与动作的耦合
3. 数据稀缺：遥操作和动捕难以规模化，是当前主要瓶颈
4. 代表任务：桌面/地面物体抓取、搬运、爬楼梯同时持物

## 代表工作

- [[GRAIL]]: 全数字化数据生成框架，覆盖 pick-up、whole-body manipulation、sitting、terrain traversal
- [[CHOIS]]: 从视频重建 HOI 轨迹用于 loco-manipulation
- [[ResMimic]]: 残差学习实现全身 loco-manipulation

## 相关概念

- [[HOI]]
- [[SONIC]]
- [[机器人操作]]
- [[sim-to-real]]
