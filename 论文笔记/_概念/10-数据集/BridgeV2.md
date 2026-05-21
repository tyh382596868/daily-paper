---
type: concept
aliases: [Bridge V2, Bridge Dataset V2]
---

# BridgeV2

## 定义

UC Berkeley 发布的大规模真机机器人操作数据集，使用 WidowX 250 6-DoF 机械臂在多种厨房 / 桌面环境下采集，包含 60k+ 操作演示，覆盖抓取、放置、按压等多种任务，是 [[VLA]] 预训练的常用数据。

## 核心要点

1. **平台**：WidowX 250（6-DoF + 夹爪）；
2. **规模**：60k+ 演示，24 个场景，13 类技能；
3. **模态**：第三人称 RGB + 本体感觉 + 自然语言指令；
4. 是 [[OpenVLA]]、[[π₀]]、[[AR-VLA]] 等 generalist 策略的预训练源；
5. 配套仿真环境是 [[SimplerEnv]]，用于离线评估。

## 代表工作

- Walke et al., "BridgeData V2: A Dataset for Robot Learning at Scale" (CoRL 2023)
- [[OpenVLA]] / [[AR-VLA]]：作为预训练数据

## 相关概念

- [[SimplerEnv]]
- [[OXE]]
- [[DROID 数据集]]
- [[VLA]]
