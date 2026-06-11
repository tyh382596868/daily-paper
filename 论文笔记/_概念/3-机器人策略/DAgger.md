---
type: concept
aliases: [Dataset Aggregation, data aggregation imitation learning]
---

# DAgger

## 定义
Dataset Aggregation 的缩写。一种在线模仿学习算法：在策略执行时收集人类/专家修正，将修正数据追加到训练集并重新训练，迭代消除 compounding errors。

## 数学形式
$$\pi_{n+1} = \text{Train}(\mathcal{D}_1 \cup \mathcal{D}_2 \cup \ldots \cup \mathcal{D}_n)$$
$$\mathcal{D}_{n} = \{(s, a^*) \mid s \sim \pi_n, a^* \sim \pi_{\text{expert}}\}$$

## 核心要点
1. 解决行为克隆（BC）的分布偏移问题：BC 在训练分布外状态上性能大幅下降
2. 每轮用当前策略收集状态，向专家查询最优动作，合并到数据集后重训
3. 缺点：需要在线专家访问（人类标注或 oracle），数据收集成本高
4. 在 VLA 后训练中，DAgger 只能间接利用失败信号，且需要持续专家参与

## 代表工作
- [[FlowPRO]]：将 DAgger 作为对比基线，提出无需专家在线参与的 offline 偏好优化替代

## 相关概念
- [[behavior cloning]]（DAgger 的基础和对比方法）
- [[VLA 后训练]]（应用场景）
- [[Compounding Errors]]（DAgger 解决的核心问题）
