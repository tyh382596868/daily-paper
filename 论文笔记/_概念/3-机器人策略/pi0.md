---
type: concept
aliases: [pi-zero, Physical Intelligence pi0, pi zero policy]
---

# pi0

## 定义

Physical Intelligence (π) 提出的基于流匹配（Flow Matching）的通用机器人策略基础模型，通过在大规模异构机器人数据上预训练，并支持任务特定微调，实现跨机器人本体的零样本/少样本泛化。

## 核心要点

1. **流匹配动作头**: 使用 Flow Matching 连续生成动作分布，优于离散化或 DDPM 采样
2. **异构数据预训练**: 在来自多种机器人平台的大规模数据集上训练，提升泛化能力
3. **VLM 骨干**: 基于视觉语言模型（VLM）处理多模态指令和观测
4. **轻量微调**: 预训练模型可通过少量任务数据高效适配特定场景

## 代表工作

- [[RoboScape]]: 使用 RoboScape 生成数据增强 pi0 的策略训练，提升操作成功率
- [[pi0.5]]: pi0 的后续版本，增强长时域规划能力

## 相关概念

- [[Diffusion Policy]]: 另一主流机器人扩散策略方法
- [[Flow Matching]]: pi0 动作生成所用的核心生成方法
- [[Action Chunking]]: pi0 采用的动作块预测策略
