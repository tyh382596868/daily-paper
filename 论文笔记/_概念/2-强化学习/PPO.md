---
type: concept
aliases: [Proximal Policy Optimization, 近端策略优化]
---

# PPO

## 定义
一种基于策略梯度的强化学习算法，通过 clip 约束限制策略更新幅度，在样本效率和训练稳定性之间取得平衡。

## 数学形式
$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$\hat{A}_t$ 为优势估计。

## 核心要点
1. clip 机制避免策略更新过大导致训练崩溃
2. 通常与 GAE（广义优势估计）配合使用
3. 是目前机器人控制领域最常用的 on-policy RL 算法之一

## 代表工作
- Schulman et al., 2017: PPO 原始论文

## 相关概念
- [[SAC]]
- [[MARL]]
- [[CEM]]
