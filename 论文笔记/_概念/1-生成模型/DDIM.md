---
type: concept
aliases: [DDIM, Denoising Diffusion Implicit Models, 去噪扩散隐式模型]
---

# DDIM (Denoising Diffusion Implicit Models)

## 定义

DDPM 的非马尔可夫确定性采样变体：在保持训练目标不变（仍用 [[去噪扩散概率模型|DDPM]] 损失训练）的前提下，把生成过程从随机采样改为可控的确定性 ODE 轨迹，从而支持**跳步采样**——通常 10–50 步即可生成高质量样本，远少于 DDPM 的 1000 步。

## 数学形式

$$
x_{t-1} = \sqrt{\alpha_{t-1}} \cdot \hat{x}_0(x_t) + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \cdot z
$$

当 $\sigma_t = 0$ 时变为确定性 DDIM。$\hat{x}_0(x_t) = (x_t - \sqrt{1-\alpha_t} \epsilon_\theta) / \sqrt{\alpha_t}$。

## 核心要点

1. **训练 = DDPM**：无需重新训练即可用于已有扩散模型
2. **确定性采样**：$\sigma_t = 0$ 时给定 $x_T$ 输出唯一
3. **跳步加速**：在 timestep 子集上采样，10–50 步即可
4. **可逆性**：支持 image editing 中的 inversion

## 代表工作

- Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- [[GAF]]: 训练 50 步、推理仅 3 步 DDIM 做动作扩散去噪

## 相关概念

- [[去噪扩散概率模型]]
- [[扩散模型]]
- [[Diffusion Policy]]
- [[Flow Matching]]
- [[采样]]
