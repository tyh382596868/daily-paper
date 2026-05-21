---
type: concept
aliases: [Mixture of Contrastive Experts, 对比专家混合, MoCE]
---

# Mixture of Contrastive Experts

## 定义
对比专家混合是分析 [[Product of Experts|对比专家乘积 (PoCE)]] 行为的理论视角——把每个专家建模为[[GMM|高斯混合]]，研究对比变换如何重加权各混合分量。

## 数学形式
专家建模为高斯混合：

$$
p_k(x) = \sum_{m=1}^{M} w_{k,m}\,\mathcal{N}(x\mid\mu_{k,m},\sigma^2_{k,m})
$$

对比加权下混合权重变换（$\rho_k=\pi_k^i/\omega_k^i$ 为对比比率）：

$$
\tilde{\pi}_k^i \propto (\omega_k^i)(\rho_k)^{\alpha_i}
$$

## 核心要点
1. **Proposition 1**：在[[核密度估计]]与不相交支撑假设下，对比变换保持高斯核形状不变，只重加权混合权重。
2. **模式选择性**：真实模式（$\rho_k\ge\tau$）被放大，虚假模式（$\rho_k\le\tau^{-1}$）被压制。
3. 玩具示例（Figure 5）显示：标准 PoE 会错放概率质量、制造虚假模式；PoCE 干净分离主导模式与虚假模式。

## 代表工作
- [[CoME]]: 用该视角证明 PoCE 不损失多样性

## 相关概念
- [[Product of Experts]]
- [[GMM]]
- [[核密度估计]]
