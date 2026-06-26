---
type: concept
aliases: [Robotics Diffusion Transformer, RDT-1B]
---

# RDT

## 定义
Robotics Diffusion Transformer，一种基于 Diffusion Transformer (DiT) 架构的 VLA 基础模型，以扩散过程生成机器人动作序列，支持多模态条件输入（视觉 + 语言 + 本体感知）。

## 核心要点
1. 采用 [[DiT]] 框架在动作空间做扩散去噪，而非 autoregressive token 预测
2. 模型规模达 1B 参数（RDT-1B），是早期大规模操作扩散模型之一
3. 支持双臂操作和多任务泛化
4. 训练数据涵盖多个 robot demo 数据集，具备跨任务/跨机器人泛化

## 数学形式
$$a_{0:H} \sim p_\theta(a | o, l) = \int p(a_T) \prod_{t=1}^{T} p_\theta(a_{t-1}|a_t, o, l) \, da_{1:T}$$

## 代表工作
- RDT-1B 原论文 (Liu et al., 2024)

## 相关概念
- [[Diffusion Policy]]
- [[Flow-Matching]]
- [[VLA]]
- [[OpenVLA]]
