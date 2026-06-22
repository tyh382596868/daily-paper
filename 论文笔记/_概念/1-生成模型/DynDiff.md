---
type: concept
aliases: [DynDiff, Dynamic Diffusion World Model]
---

# DynDiff

## 定义
用于具身 AI 的动作条件视频扩散世界模型，支持通过 RL 微调进行探索性 rollout 生成。

## 核心要点
1. 基于视频扩散（DiT 架构），以机器人动作为条件生成未来帧
2. 支持 CFG（Classifier-Free Guidance）控制动态强度
3. 配合 SAGE reward agent 实现主动探索训练

## 代表工作
- [[SAGE]]：Reward as Agent for Embodied World Models（2606.19990）

## 相关概念
- [[World Model]]
- [[Diffusion Model]]
- [[GRPO]]
- [[AGIBOT-World]]
