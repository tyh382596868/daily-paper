---
type: concept
aliases: [stop-gradient, Stop Gradient, stopgrad, sg, 停止梯度]
---

# stop-gradient

## 定义

**stop-gradient（停止梯度，常缩写 sg 或 stopgrad）** 是深度学习中常用的算子：在前向传播时表现为恒等函数 $\operatorname{sg}(x) = x$，在反向传播时把流入它的梯度截断为零。

## 数学表达

$$
\operatorname{sg}(x) = x,\quad \frac{\partial \operatorname{sg}(x)}{\partial x} = 0
$$

PyTorch 中是 `x.detach()`，JAX 中是 `jax.lax.stop_gradient(x)`，TF 中是 `tf.stop_gradient(x)`。

## 典型用途

1. **对比学习目标**：BYOL、SimSiam 中防止 target encoder 被更新导致塌缩。
2. **[[KL 散度]] 双向分解**：[[DreamerV3]] 把 $D_{KL}$ 拆成 dynamics loss 与 representation loss，各自用 sg 切断对方更新——避免先验/后验同时移动塌缩。
3. **straight-through estimator**：离散采样的近似梯度（forward 用离散值、backward 用 logits 的梯度）。
4. **Target network**：DQN / Actor-Critic 中 target 网络用 sg 防止 bootstrap 目标的梯度回流。

## 在 [[MAD]] 中的具体用途

$$
\begin{aligned}
\mathcal{L}_{dyn} &= D_{KL}\!\big[\operatorname{sg}(q_\psi(z_t)) \,\|\, p_\psi(\hat{z}_t)\big] \\
\mathcal{L}_{rep} &= D_{KL}\!\big[q_\psi(z_t) \,\|\, \operatorname{sg}(p_\psi(\hat{z}_t))\big]
\end{aligned}
$$

`dyn` 只更新先验、`rep` 只更新后验，分别独立优化。

## 关联概念

- [[KL 散度]]
- [[DreamerV3]]
- [[straight-through estimator]]
- [[Categorical Latent]]
