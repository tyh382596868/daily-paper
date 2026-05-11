---
type: concept
aliases: [Diffusion Policy, DP, 扩散策略]
---

# Diffusion Policy

## 定义

Diffusion Policy 把机器人策略 $\pi(a|o)$ 建模为**条件扩散模型**：以观测（图像 / 状态）为条件，对动作（或动作序列）做去噪生成。相比高斯 / 离散策略，能自然建模多模态、长动作序列；最早由 Chi et al. (2023) 系统提出。

## 数学形式

策略训练目标：

$$
\mathcal{L}_{DP} = \mathbb{E}_{\tau, t, a_0, \epsilon}\left[\big\|\epsilon - \epsilon_\theta(a_t, t \mid o)\big\|^2\right]
$$

其中 $a_t = \sqrt{\bar\alpha_t}\,a_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，$o$ 是观测条件。

推理时从 $a_T\sim\mathcal{N}(0,I)$ 出发，迭代去噪得到动作序列 $a_0$。

## 核心要点

1. **多模态分布**: 对同一观测可生成多种合理动作，避免高斯策略 "mode collapse"
2. **动作分块**: 通常一次预测 $H$ 步动作（[[Action Chunking]]），减少推理频次
3. **条件机制**: 观测可通过 FiLM / [[交叉注意力]] / [[AdaLN]] 注入去噪网络
4. **加速变体**: [[Flow Matching]]、Consistency Models、DDIM 等可显著减少采样步数
5. **VLA 中的位置**: 是 [[π₀]]、[[Pi05]]、[[OpenVLA]] 等 [[VLA]] 模型 action head 的常见选择之一

## 代表工作

- Chi et al., 2023: Diffusion Policy（原始论文）
- [[π₀]]: 用 [[Flow Matching]] 训练的扩散策略 head
- [[Pi05]]: π₀ 的升级版

## 相关概念

- [[Flow Matching]]
- [[扩散变换器|DiT]]
- [[Action Chunking]]
- [[VLA]]
