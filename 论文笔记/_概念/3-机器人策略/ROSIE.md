---
type: concept
aliases: [ROSIE, Robot Learning with Semantically Imagined Experience]
---

# ROSIE

## 定义

通过 text-to-image inpainting 把已有真实机器人观测中的物体/背景替换为新外观，从而**就地增广**数据的方法（Robot Learning with Semantically Imagined Experience，Yu et al. 2023）。

## 核心要点

1. **In-place 视觉增广** — 不改变机器人轨迹，只换皮
2. 优点：免重新采集，免仿真器，原始动作完全有效
3. 缺点：无法产生**新的物理配置 / 新的交互布局** — 增广多样性受原始数据限制
4. [[RoboDream]] 把 ROSIE 作为路线对照 — 强调 ROSIE 不能扩展行为分布

## 代表工作

- Yu et al. 2023: ROSIE 原始论文

## 相关概念

- [[RoboEnvision]]
- [[Compositional World Models]]
- [[图像编辑模型]]
