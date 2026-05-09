---
type: concept
aliases: [DROID, DROID Dataset]
---

# DROID 数据集

## 定义

DROID（Distributed Robot Interaction Dataset）是一个大规模真实世界机器人操作数据集，包含使用 Franka 机械臂在多种场景下采集的多样化操控轨迹，配有多视角摄像头和自然语言标注。

## 核心要点

1. 多场景多任务：覆盖厨房、办公室、家庭等真实环境，操作对象和任务多样
2. 随机摄像头位置：不同于固定视角数据集，DROID 摄像头位置随机，对泛化性要求更高
3. 大规模：原始数据超过 7.6 万条轨迹，是主流泛化操作研究的重要来源

## 代表工作

- [[MolmoAct2]]: 使用 DROID 数据训练 MolmoAct2-DROID，在 5 个新任务上达到 87.1% 成功率
- [[π0.5]]: 同样基于 DROID 数据训练的泛化操作策略

## 相关概念

- [[VLA]]
- [[Flow Matching]]
