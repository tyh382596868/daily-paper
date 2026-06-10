---
type: concept
aliases: [RoboFlamingo, Robo-Flamingo]
---

# RoboFlamingo

## 定义

基于 OpenFlamingo（视觉语言基础模型）微调的机器人操作策略模型，通过在预训练 VLM 上添加策略头并在 CALVIN 等基准上微调，实现语言条件下的机器人操控。

## 核心要点

1. **VLM 微调范式**: 在 OpenFlamingo 冻结主干上添加可训练策略头，利用 VLM 的视觉语言理解能力。
2. **CALVIN 基准**: RoboFlamingo 是 CALVIN ABC→D 基准上的早期强基线，后被 UniVLA 等方法超越。
3. **历史上下文**: 使用多帧视觉历史作为条件，增强长时序理解能力。

## 代表工作

- RoboFlamingo（Li et al., 2023）: 首次将 Flamingo 架构用于机器人操控策略学习

## 相关概念

- [[VLA]]
- [[OpenVLA]]
- [[CALVIN]]
