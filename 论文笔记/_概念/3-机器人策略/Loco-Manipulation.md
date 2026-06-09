---
type: concept
aliases: [Loco-Manipulation, loco-manipulation, 运动操作, 移动操作]
---

# Loco-Manipulation

## 定义
Loco-Manipulation 是指机器人（尤其是人形机器人）同时执行运动（Locomotion）和操作（Manipulation）的任务范式，要求机器人在移动过程中与物体或环境进行交互，是人形机器人实现通用具身智能的核心挑战。

## 核心要点
1. **运动与操作的联合控制**: 全身协调——腿部负责移动和稳定，手臂负责抓取和操作，二者需同步规划
2. **任务多样性**: 包括行走中抓取/放置物体、爬楼梯同时保持平衡、全身推拉操作、坐下/站起等
3. **数据稀缺**: 真实演示数据难以大规模获取，因此合成数据生成（如 GRAIL）是扩展的关键路径
4. **全身控制器**: 需要能够同时处理下肢运动和上肢操作的控制器，如 SONIC、WholeBodyVLA 等

## 代表工作
- [[GRAIL]]: 通过 3D 资产 + 视频基础模型生成 22k+ Loco-Manipulation 轨迹的数字化数据生成框架
- [[SONIC]]: 用于 Loco-Manipulation 的预训练全身追踪控制器
- [[ResMimic]]: 通过残差学习将人体动作迁移到人形机器人 Loco-Manipulation
- [[AHEAD]]: 人形机器人全身操作策略
- [[WholeBodyVLA]]: 基于 VLA 的全身 Loco-Manipulation 策略

## 相关概念
- [[Visuomotor Policy]]: Loco-Manipulation 的视觉运动控制实现
- [[HOI]]: 人-物交互数据，Loco-Manipulation 训练的重要来源
- [[Sim-to-Real]]: Loco-Manipulation 策略通常在仿真中训练后迁移到真机
- [[IsaacLab]]: 常用的 Loco-Manipulation 仿真训练平台
