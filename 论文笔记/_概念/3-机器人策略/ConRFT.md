---
type: concept
aliases: [ConRFT, Constrained RL Fine-Tuning]
---

# ConRFT

## 定义
VLA 模型的约束强化学习微调方法，通过约束优化防止 RL 微调过程中的策略退化（catastrophic forgetting），是 VLA RL 微调领域的主要基线之一。

## 核心要点
1. 在 RL 目标上加入行为克隆约束，防止远离初始策略
2. 对比 vanilla PPO 提供更稳定的 VLA 微调基线
3. 常与 FORCE、ROAD-VLA 等更新工作对比

## 代表工作
- [[FORCE]]: 与 ConRFT 对比，验证 value-calibrated warm-up 的效果

## 相关概念
- [[PPO]]
- [[OpenVLA]]
- [[VLA]]
