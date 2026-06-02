---
type: concept
aliases: [DROID, DROID 数据集]
---

# DROID

## 定义

DROID (Distributed Robot Interaction Dataset) 是一个跨实验室、跨场景的大规模机器人操作数据集，包含数十万条真实操作演示，由社区协作收集。

## 核心要点

1. 覆盖多实验室、多场景、多任务
2. 含语言指令 + 多视角观测 + 动作轨迹
3. [[VLA]] 模型常用的预训练 / 微调数据源
4. 相对于 [[OXE]] 更窄但质量更高

## 代表工作

- [[Pi05|π₀.₅]] 及众多 [[VLA]] 模型使用 DROID 微调
- [[RoboDream]]: 把 ~40k 条 DROID episodes 当作 "轨迹银行"，通过 retrieval & rebirth 重生为照片级合成演示

## 相关概念

- [[OXE]]
- [[VLA]]
- [[Action Chunking]]
