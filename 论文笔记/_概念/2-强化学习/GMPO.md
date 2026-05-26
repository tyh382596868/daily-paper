---
type: concept
aliases: [Geometric Mean Policy Optimization]
---

# GMPO

## 定义

[[GRPO]] 的几何平均变体,把 token 级 clipped 重要性比的连乘换成长度归一化的几何平均, 缓解长序列上梯度爆炸/消失。

## 数学形式

$$
\mathcal{J}_{GMPO} = \mathbb{E}\!\left[\big|\min\!\big(r_{i,t}(\Theta) \hat{A}_i,\; \mathrm{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\big)\big|^{1/|o_i|} \cdot \mathrm{sign}(\hat{A}_i) \right]
$$

其中 $|o_i|$ 是 rollout $i$ 的 token 数, 指数 $1/|o_i|$ 实现长度归一化(几何平均效应)。

## 核心要点

1. **与 GRPO 的差异**: GRPO 用 token 级期望(算术平均), GMPO 用几何平均, 对长序列更稳定。
2. **符号处理**: 取绝对值做几何平均, 再乘 $\mathrm{sign}(\hat{A}_i)$ 保留优势方向。
3. **同样不需要 critic**: 优势 $\hat{A}_i$ 仍用组内归一化 reward。
4. **后训练敏感**: 用 [[Muon]] 训练时同样崩溃, 用 [[Pion]] 训练稳定。

## 代表工作

- [[Pion]]: 在 GMPO + Qwen3-1.7B/4B 上验证 Pion 的稳定性

## 相关概念

- [[GRPO]]
- [[PPO]]
- [[Pion]]
- [[梯度信噪比]]
