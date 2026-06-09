---
type: concept
aliases: [双足运动, 步态控制, legged locomotion, bipedal locomotion]
---

# locomotion

## 定义

locomotion（运动/步态控制）是机器人学中控制腿式机器人（四足/双足）稳定行走、奔跑、在复杂地形中移动的技术方向，是人形机器人 [[全身控制]] 的重要组成部分。

## 核心要点

1. **平衡与稳定性**：维持机器人动态平衡，防止跌倒
2. **速度跟踪**：跟踪期望底盘速度命令（前后/横移/转向）
3. **地形适应**：在平坦地形、台阶、斜坡等不同地形泛化
4. **与操作耦合**：在人形机器人 loco-manipulation 场景中，locomotion 与手臂操作相互影响，需全身协调

## 代表工作

- [[HANDOFF]]: 专门训练 15-DoF Locomotion Teacher（下半身），通过速度 context 与 WBC Teacher 做凸组合蒸馏

## 相关概念

- [[全身控制]]
- [[PPO]]
- [[Unitree G1]]
