---
type: concept
aliases: [Cramér-Wold theorem, Cramer-Wold theorem, Cramer–Wold]
---

# Cramér-Wold 定理

## 定义

测度论中的经典定理：$\mathbb{R}^d$ 上两个概率分布相同当且仅当它们在**所有方向**上的一维投影分布都相同。等价地，多元随机向量 $\mathbf{X}$ 的分布完全由其所有线性投影 $\mathbf{u}^\top \mathbf{X}$（$\mathbf{u} \in S^{d-1}$）的分布族决定。

## 数学形式

$$
\mathbb{P}_{\mathbf{X}} = \mathbb{P}_{\mathbf{Y}} \quad \Longleftrightarrow \quad \forall \mathbf{u} \in S^{d-1},\; \mathbb{P}_{\mathbf{u}^\top \mathbf{X}} = \mathbb{P}_{\mathbf{u}^\top \mathbf{Y}}
$$

特别地，$\mathbf{X} \sim \mathcal{N}(0, \mathbf{I})$ 当且仅当对所有单位向量 $\mathbf{u}$，$\mathbf{u}^\top \mathbf{X} \sim \mathcal{N}(0, 1)$。

## 核心要点

1. **降维利器**：把 $d$ 维分布检验降到无穷多个 1D 检验，再用蒙特卡洛抽 $M$ 个方向近似
2. **可微替代**：在自监督表示学习中可作为分布匹配损失的理论基础
3. **与 Sliced Wasserstein 同源**：都利用"投影到 1D"的思路绕过高维分布距离的难题

## 代表工作

- [[SIGReg]] / [[LeWM]]: 用该定理把"latent 是否高斯"问题降为多个 1D 正态性检验
- Sliced Wasserstein Distance：相同思路在生成模型评估中的应用

## 相关概念

- [[SIGReg]]
- [[Epps-Pulley 检验]]
- [[特征函数]]
