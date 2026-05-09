---
type: concept
aliases: [DROID, DROID dataset]
---

# DROID 数据集

## 定义

DROID 是一个大规模 Franka 机器人操控数据集，包含 7.46 万条遥操作轨迹，是目前最大的开放机器人数据集之一。

## 核心要点

1. 主要使用 Franka Panda 机器人收集
2. 包含多种操控任务（抓取、放置、开合等）
3. MolmoAct2 对其进行质量筛选后用于预训练

## 代表工作

- [[MolmoAct2]]：MolmoAct2-DROID 在其上训练，真实操控任务 87.1%

## 相关概念

- [[VLA]]
- [[OpenFAST]]
