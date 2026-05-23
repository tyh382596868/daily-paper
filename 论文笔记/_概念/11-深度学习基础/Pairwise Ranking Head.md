---
type: concept
aliases: [Pairwise Ranking Head, 成对排序头, Pairwise Head]
---

# Pairwise Ranking Head

## 定义

接受两个表征 $z_i, z_j$ 输出一个标量距离 / 排序分数的小型网络模块，用于学习「相对距离」而非绝对值。在 [[TRM]] 中用作 [[Latent MPC]] 的终端代价。

## 数学形式

特征图:
$$
\psi(z_i, z_j) = [z_i, z_j, z_i - z_j, |z_i - z_j|]
$$

前向 (两层 MLP):
$$
m_\phi(z_i, z_j) = \mathrm{Softplus}\big(W_3 \, \sigma(W_2 \, \sigma(W_1 \psi + b_1) + b_2) + b_3\big)
$$

## 核心要点

1. **特征对称性**: 同时含 $z_i-z_j$ 和 $|z_i-z_j|$，方便头学到方向无关的距离
2. **非负输出**: 输出层用 [[Softplus]] 保证 $m_\phi \ge 0$，符合"距离"语义
3. **轻量**: 典型配置 2 层 256-unit MLP（~200K 参数），训练时间几分钟
4. **TRM 应用**: 用「同轨迹两点时间间隔 $|t_i-t_j|$」作为监督标签，让 $m_\phi$ 学到「可达性距离」

## 代表工作

- [[TRM]]: 用作 [[Latent MPC]] 的替换 / 混合终端代价
- 一般 [[Metric Learning]] 文献中的 siamese / triplet 头

## 相关概念

- [[TRM]]
- [[SiLU]]
- [[Softplus]]
- [[Smooth-L1 Loss]]
- [[Horizon-Matched Supervision]]
