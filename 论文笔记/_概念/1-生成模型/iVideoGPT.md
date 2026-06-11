---
type: concept
aliases: [interactive VideoGPT, iVideoGPT]
---

# iVideoGPT

## 定义
基于自回归 Transformer 的交互式视频预测模型，支持动作条件的视频生成，用于具身 AI 的世界模型构建和策略评估。

## 数学形式

$$
p(v_{t+1} | v_{1:t}, a_{1:t}) = \text{Transformer}(v_{1:t}, a_{1:t})
$$

## 核心要点
1. 动作条件视频预测，适合机器人操作场景的世界模型
2. 基于 GPT 风格自回归建模，按帧逐步预测
3. 可作为策略评估器，但与真实仿真器相关性较低（相比 RoboScape）

## 代表工作
- [[RoboScape]]: 与 iVideoGPT 对比，作为世界模型基线

## 相关概念
- [[Action-Conditioned World Model]]
- [[自回归视频生成]]
- [[具身世界模型]]
