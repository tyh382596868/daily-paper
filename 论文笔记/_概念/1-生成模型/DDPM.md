---
type: concept
aliases: [Denoising Diffusion Probabilistic Model, 去噪扩散概率模型]
---

# DDPM

## 定义

DDPM（Denoising Diffusion Probabilistic Models）是一类基于马尔可夫链的生成模型，通过逐步向数据添加高斯噪声（前向过程）再学习逐步去噪（反向过程）来建模数据分布。

## 数学形式

**训练目标（去噪得分匹配简化形式）**:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

## 核心要点

1. **前向过程**: 固定马尔可夫链，逐步向数据加噪，$T$ 步后趋近标准正态分布
2. **反向过程**: 学习参数化的去噪转移核 $p_\theta(x_{t-1}|x_t)$
3. **简化训练目标**: 预测噪声 $\epsilon$ 等价于预测得分函数，且实践效果更好
4. **推理速度**: 标准 DDPM 需要 1000 步推理，后续工作（DDIM、DPM-Solver）大幅加速

## 代表工作

- [[LAD]]: 在隐动作空间上使用扩散去噪目标训练跨具身操控策略
- [[Diffusion Policy]]: 将 DDPM 应用于机器人动作预测

## 相关概念

- [[Diffusion Policy]]
- [[Diffusion Model]]
- [[对比学习]]
