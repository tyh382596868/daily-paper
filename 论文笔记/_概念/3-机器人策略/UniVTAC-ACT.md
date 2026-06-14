---
type: concept
aliases: [UniVTAC ACT, Universal Tactile ACT]
---

# UniVTAC-ACT

## 定义

UniVTAC-ACT 是 UniVTAC 基准配套的 [[ACT (Action Chunking Transformer)|ACT]] 策略基线，针对仿真多任务触觉操作环境（UniVTAC）中的接触丰富任务而设计，是触觉操作策略的标准对比方法之一。

## 核心要点

1. 基于 [[ACT (Action Chunking Transformer)|ACT]]（Action Chunking Transformer）框架，结合触觉观测输入
2. 在 UniVTAC 仿真基准上训练和评估，涵盖 Lift、Insert、Put 等接触丰富任务
3. 传感器特定训练，不具备跨传感器迁移能力
4. 在 UniVTAC 基准上平均成功率约 43%，被 [[FTP-1]] 大幅超越（66.66%）

## 代表工作

- [[FTP-1]]: 在 UniVTAC 基准上将 UniVTAC-ACT 作为主要对比 baseline（arXiv:2602.10093 Chen et al., 2026）

## 相关概念

- [[ACT (Action Chunking Transformer)]]
- [[MTTS]]
- [[Tactile-VLA]]
- [[VITaL]]
