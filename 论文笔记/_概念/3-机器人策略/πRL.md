---
type: concept
aliases: [piRL, π_RL, pi-RL, Online RL for VLA]
---

# πRL

## 定义

针对基于 Flow/Diffusion 的 VLA 模型（如 π0）的在线强化学习微调框架，通过在真实或仿真环境中收集奖励信号并用 RL 优化策略，突破纯模仿学习的性能上限。

## 核心要点

1. **目标**: 解决 Flow-based VLA 的离线 BC 训练无法从环境反馈中学习的问题。
2. **流程**: VLA 策略在环境中执行，收集奖励 → RL 梯度回传更新策略网络参数。
3. **挑战**: Flow 模型推理是多步去噪过程，如何高效计算 RL 梯度是关键技术问题。
4. **与 RLPD 关系**: 都结合了先验数据与在线学习，但 πRL 专注于 Flow/Diffusion 架构的 VLA。

## 代表工作

- πRL（2025）: arXiv:2510.25889，"Online RL Fine-tuning for Flow-based Vision-Language-Action Models"

## 相关概念

- [[pi0]]
- [[RLPD]]
- [[VLA 后训练]]
- [[强化学习]]
