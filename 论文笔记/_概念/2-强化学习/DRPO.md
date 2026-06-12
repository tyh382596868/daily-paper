---
type: concept
aliases: [Divergence-Regularized Policy Optimization]
---

# DRPO

## 定义
LLM RL 中重新思考散度正则化的框架，提出更合理的 off-policy 散度约束方案。

## 数学形式
$$\mathcal{L}_\text{DRPO} = \mathbb{E}[r(y)] - \beta D_\text{TV}(\pi || \pi_\text{ref})$$

## 核心要点
1. 针对 LLM RL 中 training-inference mismatch 导致的 off-policy 问题
2. 用 Total Variation 散度代替 KL 散度做正则化
3. 与 GRPO、PPO、DPO 对比

## 相关概念
- [[GRPO]]
- [[PPO]]
