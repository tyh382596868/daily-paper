---
type: concept
aliases: [EDM Noise Weighting, EDM Framework, EDM 框架]
---

# EDM 噪声加权

## 定义

EDM（**E**lucidating the **D**esign space of diffusion-based generative **M**odels，Karras et al. 2022）噪声加权框架是一种针对扩散模型训练的损失权重设计：将每个去噪步的损失按当前噪声水平 $\sigma$ 自适应调制，使得不同 SNR（信噪比）下的训练信号都能对最终性能贡献相近，避免高噪声步主导梯度。

## 数学形式

$$
\lambda(\sigma) = \frac{\sigma^2 + \sigma_{data}^2}{(\sigma \cdot \sigma_{data})^2}
$$

或在简化的等价形式中：

$$
w(\sigma) = \frac{1}{\sigma^2 + \sigma_{data}^2}
$$

其中 $\sigma$ 为当前去噪步的噪声标准差，$\sigma_{data}$ 通常取 0.5（归一化图像数据）。

## 核心要点

1. **SNR 均衡**: 让训练信号在所有噪声水平上均衡，避免极小或极大噪声步主导优化
2. **替代 DDPM 经验权重**: 比 DDPM 的 $\alpha_t, \beta_t$ 调度更原则化、更灵活
3. **跨任务通用**: 不仅适用于像素扩散，也用于潜空间扩散（[[LDM]]、[[DiT]]）与 [[Flow Matching]] 变体
4. **辅助任务的权重调制**: 多任务联合训练时（如 [[状态感知视频世界模拟器]] 中的视频+奖励双目标），可用 $\lambda(\sigma)$ 让辅助任务在不同噪声下贡献合理

## 在多目标训练中的应用

[[World-VLA-Loop]] 用 EDM 权重调制视频损失与奖励损失的混合比例：噪声小（接近真实图像）时给奖励项更大权重，噪声大时主要让视频生成主导训练。

## 代表工作

- Karras et al. 2022 "Elucidating the Design Space of Diffusion-based Generative Models"
- [[World-VLA-Loop]]: 用 EDM 框架调制视频+奖励联合损失权重

## 相关概念

- [[Flow Matching]]
- [[DDIM]]
- [[Score Function]]
- [[联合训练目标]]
- [[状态感知视频世界模拟器]]
