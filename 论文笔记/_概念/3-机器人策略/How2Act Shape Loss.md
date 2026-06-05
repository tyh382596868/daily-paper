---
type: concept
aliases: [How2Act Shape, 形状扩散损失]
---

# How2Act Shape Loss

## 定义

AffordanceVLA 中用于生成目标物体 3D 形状 latent 的条件扩散损失，由 shape token 作为条件。

## 数学形式

$$
\mathcal{L}_{shape} = \mathbb{E}_{t, \varepsilon} \left[ \| \varepsilon - \hat{\varepsilon}_\theta(x_t, t, \bar{h}_{shape}) \|^2 \right]
$$

## 核心要点

1. **条件扩散**: 以 Affordance Expert 输出的 shape token 为条件，[[Denoising Loss|噪声预测损失]]。
2. **生成性**: 扩散适合捕捉形状的多模态分布。
3. **与 Layout 互补**: Shape 给"是什么形状"，Layout 给"在哪、什么朝向"。
4. **绝对精度有限**: 作者承认 shape token accuracy 相对低，但仍对控制有效。

## 代表工作

- [[AffordanceVLA]]

## 相关概念

- [[How2Act Layout Loss]]
- [[Diffusion Model]]
- [[Denoising Loss]]
- [[Affordance Generation Expert]]
- [[Affordance Forecasting]]
