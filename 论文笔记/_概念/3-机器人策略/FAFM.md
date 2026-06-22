---
type: concept
aliases: [Frequency-Aware Flow Matching, 频率感知流匹配]
---

# FAFM

## 定义

FAFM（Frequency-Aware Flow Matching）是一种零参数 plug-in 机器人动作生成方法，将动作 chunk 投影到 DCT 频域执行 Flow Matching，并施加解析一阶导数正则化，同时解决异构控制频率和时序抖动问题。

## 数学形式

联合训练损失：

$$
\mathcal{L}_{FAFM} = \mathcal{L}_{FM}^{coeff} + \lambda \cdot \mathcal{L}_{vel}, \quad \lambda = 1
$$

等价于 H¹ Sobolev 加权范数投影误差：

$$
\mathcal{L}_{FAFM}(\theta) = \left\|\hat{\xi}_\theta - P_M\xi^*\right\|_{H_\mu^1}^2 = \sum_{j=0}^{M}(1 + \lambda\omega_j^2)\left|\langle\hat{\xi}_\theta - P_M\xi^*, \phi_j\rangle_{L^2}\right|^2
$$

## 核心要点

1. **频率解耦**: DCT 系数仅依赖物理时间 $\tau = n/f_\xi$，与控制频率 $f_\xi$ 解耦，解决可识别性失败问题
2. **解析导数**: M 阶余弦子空间内速度场可解析计算（$\hat{\dot{v}}(\tau) = -\sum_j \hat{c}_j \omega_j \sin(\omega_j\tau)$），无需数值差分
3. **H¹ 理论保证**: 联合损失等价于 Sobolev H¹ 投影，高频分量受 $\omega_j^2$ 二次惩罚
4. **零参数**: 不增加网络参数，直接替换训练目标，兼容任意 Flow Matching 框架和 VLA

## 代表工作

- [[FAFM 论文]]（Guo et al., 2026）: 方法原始论文

## 相关概念

- [[Flow Matching]]
- [[离散余弦变换 (DCT)]]
- [[Sobolev 范数]]
- [[Action Chunking]]
- [[DCT 动作 Token 化]]
