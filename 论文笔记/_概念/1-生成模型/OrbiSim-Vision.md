---
type: concept
aliases: [OrbiSim Vision, OrbiSim 视觉模块, 状态引导潜空间扩散渲染]
---

# OrbiSim-Vision

## 定义

OrbiSim 的视觉渲染模块，本质是**物理状态条件化的 [[LDM|Latent Diffusion]] 模型**，把 [[OrbiSim-Dynamics]] 输出的物理状态 $\hat{x}_t$ 和静态资产 $\bar{x}$ 转换为 RGB 帧。

## 数学形式

$$
\hat{o}_t \sim p_\phi^{vis}(o_t \mid o_{t-K:t-1}, \hat{x}_t, \bar{x})
$$

训练目标（EDM 风格）：

$$
\mathcal{L}_{vis} = \mathbb{E}_{\sigma, \varepsilon}\left[ w(\sigma) \| D_\phi(y_{(0)} + \sigma\varepsilon; c_t)_t - y_{t,(0)} \|_2^2 \right]
$$

## 核心要点

1. **四种条件注入机制**:
   - 空间条件图：物体姿态投影成 heatmap，concat 到 [[UNet]]
   - 物体级 token：物体状态经 [[交叉注意力]] 注入
   - 视觉锚定：最近 $K \in \{0,\ldots,6\}$ 帧扰动 latent 拼接
   - 物理 grounding：编码 $\bar{x}$、$\hat{x}_t$
2. **训练时随机 K**: 让模型同时支持冷启动与上下文延续
3. **与 Dynamics 解耦**: 策略梯度**不经过** Vision，避开扩散链反传不稳

## 代表工作

- [[OrbiSim]]: 提出并训练该模块

## 相关概念

- [[OrbiSim-Dynamics]]
- [[LDM]]
- [[UNet]]
- [[扩散变换器]]
