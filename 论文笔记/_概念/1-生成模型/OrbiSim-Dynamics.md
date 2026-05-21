---
type: concept
aliases: [OrbiSim Dynamics, OrbiSim 动力学模块, 物体中心循环动力学]
---

# OrbiSim-Dynamics

## 定义

OrbiSim 的物理状态预测主模块，是一个**物体中心 (object-centric) 的循环 Transformer 动力学网络**，把动作、物理属性、历史状态映射为下一步物理状态预测，全程可微以支持解析策略梯度。

## 数学形式

$$
\hat{x}_t = f_\phi^{dyn}(x_{0:t-1}, a_{t-1}, \bar{x})
$$

分解为五个组件：

$$
\begin{aligned}
e_t &= f_\phi^{cp}(z_{t-1}, a_{t-1}, \bar{x}) \quad \text{(Coupling)} \\
h_t &= f_\phi^{rec}(h_{t-1}, e_t) \quad \text{(Recurrent)} \\
\hat{z}_t &= f_\phi^{tra}(h_t) \quad \text{(Transition)} \\
z_t &= f_\phi^{enc}(h_t, x_t) \quad \text{(Encoder)} \\
\hat{x}_t &= f_\phi^{dec}(\hat{z}_t) \quad \text{(Decoder)}
\end{aligned}
$$

## 核心要点

1. **物体中心表示**: 每个物体一个 token，机器人也作为 token，便于扩展到多物体
2. **Transformer Coupling Module**: 用 self-attention 显式建模多实体相互作用
3. **[[AdaLN]] 调制**: 动作和物理属性通过 scale/shift 注入，不增加 token 数量
4. **RSSM 结构**: 与 [[DreamerV3]] 类似的循环潜在状态空间，但目标是物理状态预测而非 latent imagination
5. **三项训练损失**: transition / encoding (双向带 stop-gradient) + decoder 重建

## 代表工作

- [[OrbiSim]]: 提出该模块，作为可微物理引擎的核心

## 相关概念

- [[OrbiSim-Vision]]: 与之解耦的视觉渲染模块
- [[循环状态空间模型]]
- [[AdaLN]]
- [[解析策略梯度]]
- [[世界模型]]
