---
type: concept
aliases: [Which2Act, Which2Act 重建损失]
---

# Which2Act Loss

## 定义

AffordanceVLA 中用于学习"应该作用在哪个对象"的损失：把目标对象区域用冻结 [[FLUX]] [[3D Causal VAE|VAE Encoder]] 编码成连续 latent，让 Affordance Expert 预测该 latent，使用 MSE 监督。

## 数学形式

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum \|\hat{z} - z_q\|^2
$$

## 核心要点

1. **连续 latent**: 不用离散 codebook，避免 [[VQ-VAE]] 量化误差。
2. **冻结教师**: $z_q$ 由 frozen Flux VAE Encoder 给出，作为知识蒸馏目标。
3. **对象级 grounding**: 让模型先回答 "which object"，是 Where/How 的前置。
4. **消融影响**: 去掉 Which2Act → LIBERO 95.8 → 94.6（−1.2）。

## 代表工作

- [[AffordanceVLA]]

## 相关概念

- [[Affordance Forecasting]]
- [[Affordance Generation Expert]]
- [[FLUX]]
- [[3D Causal VAE]]
- [[Where2Act Loss]]
- [[How2Act Shape Loss]]
