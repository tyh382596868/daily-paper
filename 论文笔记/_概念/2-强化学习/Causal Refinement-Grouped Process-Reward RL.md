---
type: concept
aliases: [CRG-PRL, 因果精炼分组过程奖励RL]
---

# Causal Refinement-Grouped Process-Reward RL

## 定义
PearlVLA 提出的强化学习方法，通过对同一精炼状态下的多个潜在计划编辑进行组内相对比较（group-relative advantage），用更长视野的想象未来奖励优化迭代精炼轨迹，无需学习价值函数。

## 数学形式

**组相对优势**:

$$
A^{(k)}_i = r^{(k)}_i - \frac{1}{G}\sum_{j=1}^{G} r^{(k)}_j
$$

**过程奖励**: 将第 $k$ 轮精炼后的潜在滚动解码为想象未来帧并评分

## 核心要点
1. **因果隔离**: 仅比较同一精炼状态下的候选编辑，避免不同状态难度混淆奖励信号
2. **无价值函数**: 用组内均值作基线，无需学习价值网络（类比 [[GRPO]]）
3. **过程奖励（Process Reward）**: 每轮精炼都有独立奖励信号，而非仅看终点结果
4. **longer-horizon 想象**: 用自回归扩展的想象未来帧（而非单步预测）评分，激励长程规划

## 代表工作
- [[PearlVLA]]: CRG-PRL 的提出论文，将 LIBERO 平均从 98.5% 提升至 98.7%

## 相关概念
- [[GRPO]]: 组相对策略优化，CRG-PRL 的相关方法
- [[Latent Plan Refinement]]
- [[Latent World Model]]
