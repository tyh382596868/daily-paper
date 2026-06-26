---
type: concept
aliases: [Robo-Monkey, test-time scaling robotics]
---

# RoboMonkey

## 定义
一种面向机器人操作任务的 test-time scaling 框架，通过在推理时生成多个候选动作序列并用 verifier 进行筛选，提升 VLA 策略的成功率。

## 核心要点
1. 在推理时 sample 多条候选轨迹（best-of-N），用 reward/verifier 模型打分
2. 相当于将 LLM test-time scaling 范式（如 MCTS、BoN）迁移到 embodied 任务
3. 无需重新训练基础策略，可叠加在任何 VLA 上
4. 代价是推理时间增加（N 倍 forward pass）

## 代表工作
- RoboMonkey 原论文 (2024/2025)

## 相关概念
- [[VLA]]
- [[CoT]]
- [[E-TTS]]
