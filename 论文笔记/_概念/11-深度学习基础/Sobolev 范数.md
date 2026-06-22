---
type: concept
aliases: [H¹ Sobolev Norm, Sobolev Norm, 索博列夫范数, H1 范数]
---

# Sobolev 范数

## 定义

Sobolev 范数（H¹）是一种同时衡量函数本身及其一阶导数的 L² 误差的加权范数，在函数逼近和机器学习中用于约束预测函数的平滑性。

## 数学形式

H¹ Sobolev 加权范数定义为：

$$
\|f\|_{H_\mu^1}^2 = \sum_{j=0}^{M} \mu_j \left|\langle f, \phi_j \rangle_{L^2}\right|^2
$$

其中 $\mu_j = 1 + \lambda \omega_j^2$，$\phi_j$ 为正交基函数（如余弦基），$\omega_j$ 为第 $j$ 阶频率。

当 $\lambda > 0$ 时，高频分量受 $\omega_j^2$ 二次惩罚，天然压制高频振荡。

## 核心要点

1. **频率相关权重**: 高频分量 $j$ 的权重 $\mu_j = 1 + \lambda\omega_j^2$ 随频率二次增长，有效抑制抖动
2. **平滑性保证**: 最小化 H¹ 范数等价于同时最小化函数值误差和导数误差
3. **与 L² 的关系**: 当 $\lambda=0$ 时退化为标准 L²（Sobolev H⁰）范数
4. **FAFM 理论联系**: FAFM 的联合损失 $\mathcal{L}_{FM} + \lambda\mathcal{L}_{vel}$ 严格等价于 H¹ Sobolev 投影误差（定理 1）

## 代表工作

- [[FAFM]]（Guo et al., 2026）: 证明频域 Flow Matching + 速度正则化等价于 H¹ Sobolev 范数优化

## 相关概念

- [[Flow Matching]]
- [[离散余弦变换 (DCT)]]
- [[Action Chunking]]
