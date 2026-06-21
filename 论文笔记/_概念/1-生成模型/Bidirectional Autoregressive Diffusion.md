---
type: concept
aliases: [双向自回归扩散]
---

# Bidirectional Autoregressive Diffusion

## 定义
在自回归视频生成中，同时利用前向历史帧和后向未来约束来生成当前帧，提升长视频时序一致性。

## 数学形式
$$p(x_t | x_{<t}, x_{>t}) \propto p(x_t | x_{<t}) \cdot p(x_{>t} | x_t)$$

## 核心要点
1. 不同于纯因果自回归，允许后向信息流
2. 在长视频生成中减少帧漂移
3. 与 [[Diffusion Forcing]] 的思想相关

## 代表工作
- [[BiWM]]: 采用双向自回归扩散的开源交互世界模型
- [[Diffusion Forcing]]: 扩散模型时序双向约束早期工作

## 相关概念
