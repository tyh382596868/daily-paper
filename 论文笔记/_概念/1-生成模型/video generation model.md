---
type: concept
aliases: [video generation model, 视频生成模型, VGM]
---

# Video Generation Model（视频生成模型）

## 定义
通过学习视频数据的时空分布，能够根据条件（文本、图像、动作等）生成连贯、真实视频序列的生成模型，通常基于扩散模型或自回归框架。

## 数学形式
$$p(v_{1:T} | c) = \prod_{t=1}^{T} p(v_t | v_{1:t-1}, c)$$

或基于 diffusion 的形式：$v_0 \sim p_\theta(v_0|c) = \int p_\theta(v_{0:T}|c)dv_{1:T}$

## 核心要点
1. 主要范式：视频扩散模型（如 CogVideoX、HunyuanVideo）和自回归视频生成
2. 在机器人学习中用作世界模型：给定动作序列生成未来视频帧
3. 冻结的视频生成模型可作为高层视觉规划器（zero-shot 泛化）
4. 视觉质量指标（FVD/FID）与物理可执行性不直接相关（[[Dream-exe]] 的核心发现）

## 代表工作
- [[RoboDream]]: 用视频扩散模型合成机器人演示数据
- [[VICX]]: 用冻结视频生成模型做高层视觉规划
- [[Dream-exe]]: 评估视频生成模型的物理可执行性

## 相关概念
- [[Diffusion Model]]
- [[world model]]
- [[VICX]]
