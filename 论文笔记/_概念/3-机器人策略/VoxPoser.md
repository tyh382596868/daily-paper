---
type: concept
aliases: [VoxPoser, 体素化规划器]
---

# VoxPoser

## 定义

VoxPoser 是一种基于大语言模型的零样本机器人操作规划方法，通过将语言指令转化为 3D 价值地图（value map），引导机器人末端执行器在体素空间中规划路径，无需任务特定训练数据。

## 核心要点

1. **零样本规划**: 利用 LLM 生成代码合成 3D 价值地图，无需任务演示
2. **体素空间表示**: 在三维体素网格上定义可达性和避障约束
3. **局限性**: 在需要精细操作（如 Play_jenga）或物体理解（如 Close_box）的任务上成功率为 0

## 代表工作

- Huang et al. "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models" (CoRL 2023)

## 相关概念

- [[VLA]]
- [[3DAgent]]
