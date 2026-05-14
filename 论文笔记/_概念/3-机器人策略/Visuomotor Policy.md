---
type: concept
aliases: [视觉运动策略, Visuomotor]
---

# Visuomotor Policy

## 定义

直接从视觉观测（与可选的本体感知、语言指令）映射到机器人动作的策略，不依赖大规模预训练 VLM backbone — 这是与 [[VLA]] 的主要区别。

## 数学形式

$$
p(a_{t+1:t+k} \mid o_t,\ l)
$$

## 核心要点

1. 通常由 CNN/ViT + Transformer/Diffusion head 组成，规模相对小
2. 比 [[VLA]] 更轻量，部署成本低；但语言理解和泛化能力弱
3. [[Diffusion Policy]] 是当前最具代表性的 visuomotor 实现
4. 仍然可以与[[世界模型]]耦合（[[RobotWM-Survey]] Section 2.2.1）

## 代表工作

- [[Diffusion Policy]]: 扩散模型 + 视觉编码器
- ACT (Action Chunking Transformer)
- 行为克隆（BC）基线

## 相关概念

- [[策略]]
- [[VLA]]
- [[Diffusion Policy]]
- [[RobotWM-Survey]]
