---
type: concept
aliases: [RLPD, RL with Prior Data, 先验数据强化学习]
---

# RLPD

## 定义

Reinforcement Learning with Prior Data 的缩写，一种将离线演示数据与在线强化学习相结合的方法：在 RL 训练的 replay buffer 中混入先验演示数据，加速策略学习和提升样本效率。

## 核心要点

1. **混合 Replay Buffer**: 在线 RL 收集的数据与离线演示数据按比例混合采样，两者权重可调。
2. **样本效率**: 先验数据提供初始良好行为覆盖，减少 RL 探索阶段的随机游走。
3. **适用场景**: 机器人操作中有少量专家演示但希望用 RL 进一步优化策略时特别有效。
4. **与 BC+RL 对比**: RLPD 不需要专门的预训练阶段，先验数据直接参与 Q 函数和策略更新。

## 代表工作

- Ball et al., 2023: "Efficient Online Reinforcement Learning with Offline Data"

## 相关概念

- [[强化学习]]
- [[模仿学习]]
- [[πRL]]
