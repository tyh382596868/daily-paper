---
type: concept
aliases: [HiL-RL, 人类在环强化学习]
---

# Human-in-the-Loop RL

## 定义
将人类反馈或干预嵌入强化学习训练循环，通过人类指导改善策略质量并降低样本复杂度。

## 数学形式
$$r_t = r_{env}(s_t, a_t) + \lambda \cdot r_{human}(s_t, a_t)$$

## 核心要点
1. 人类在关键决策点可以直接干预动作选择
2. 人类干预数据作为额外监督信号
3. 显著减少危险探索行为

## 代表工作
- [[UniIntervene]]: 提出 Agentic 干预的 HiL-RL 框架

## 相关概念
