---
type: concept
aliases: [SIGReg, Sketched Isotropic Gaussian Regularization]
---

# SIGReg

## 定义

SIGReg 是一种基于 Cramér–Wold 定理的**分布匹配正则项**：通过把高维潜表示投影到大量随机一维方向，并对每个一维投影做 Epps–Pulley 正态性检验，强制潜分布逼近各向同性高斯，从而防止表示坍缩（collapse）。最早提出于 LeJEPA（Balestriero & LeCun, 2025）。

## 数学形式

$$
\text{SIGReg}(Z) \triangleq \frac{1}{M}\sum_{m=1}^{M} T(Z\cdot u^{(m)}),\quad u^{(m)}\sim \text{Uniform}(S^{d-1})
$$

其中 $T(\cdot)$ 是 Epps–Pulley 一维正态性检验统计量；理论保证：

$$
\text{SIGReg}(Z)\to 0 \;\Longleftrightarrow\; P_Z\to \mathcal{N}(0,I)
$$

## 核心要点

1. **Cramér–Wold**: 匹配所有一维投影 ⇔ 匹配联合分布；用大量随机投影近似
2. **Epps–Pulley 检验**: 经验特征函数与标准正态特征函数的加权 $\ell_2$ 距离
3. **单超参**: 投影数 $M$ 不敏感，只有正则权重 $\lambda$ 需调
4. **替代 EMA / stop-gradient**: 提供形式化的防 collapse 保证，无需启发式技巧

## 代表工作

- LeJEPA (Balestriero & LeCun, 2025): 首次提出 SIGReg 用于自监督表示学习
- [[LeWM]]: 把 SIGReg 引入动作条件世界模型，实现端到端 JEPA 训练

## 相关概念

- [[JEPA]]
- [[EMA]]
