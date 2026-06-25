---
type: concept
aliases: [Cosmos World Model, NVIDIA Cosmos, Cosmos 3]
---

# Cosmos

## 定义
NVIDIA 发布的 physical AI 世界基础模型系列，支持 action-conditioned 视频生成和交互式世界模型，用于机器人和自动驾驶的 pre-training 和 sim-to-real 数据生成。

## 核心要点
1. **Cosmos 1/2**: 早期版本，以视频生成为主
2. **Cosmos 3（2026）**: omnimodal world foundation model，支持 action-conditioned 生成，直接面向 physical AI 应用
3. 与 [[Causal-rCM]] 结合：用 TF+SF 蒸馏框架实现 2-step 实时 action-conditioned 世界模型
4. 应用：机器人 sim-to-real 数据合成、交互式世界模型评测

## 代表工作
- [[Causal-rCM]]: 用 Causal-rCM 蒸馏 Cosmos 3 实现实时 interactive world model

## 相关概念
- [[Self-Forcing]]
- [[rCM]]
- [[Action-Conditioned World Model]]
