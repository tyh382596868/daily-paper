---
type: concept
aliases: [可供性, affordance, Affordances]
---

# Affordance（可供性）

## 定义

可供性指环境/物体对智能体提供的"可执行操作"的属性，是连接感知与动作的关键中间表征。例如把手"可被握住和拉动"、按钮"可被按下"，本质上回答 "what action can be done where"。

## 核心要点

1. **任务导向中间表征**: 把感知到动作的端到端映射拆为 perception → affordance → action。
2. **形式多样**: 2D 热力图（可交互像素）、3D 接触点、力旋量、6D 抓取位姿、自然语言描述等。
3. **与指令耦合**: 同一物体在不同指令下可供性不同（"open" vs "wipe" 抽屉）。
4. **可学习**: 现代方法用监督/自监督从人类交互数据（如 [[AGD20K]]）中学习可供性预测。

## 代表工作

- [[AffordanceVLA]]: 提出 Which/Where/How 三层结构化可供性作为 VLA 中间表征
- [[Where2Act 系列]]: 经典 2D 可供性图预测路线

## 相关概念

- [[Affordance Forecasting]]
- [[Which2Act Loss]] / [[Where2Act Loss]] / [[How2Act Shape Loss]] / [[How2Act Layout Loss]]
- [[AGD20K]]
- [[VLA]]
