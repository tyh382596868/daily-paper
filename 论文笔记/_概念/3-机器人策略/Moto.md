---
type: concept
aliases: [Moto-GPT, Motion Token, 运动 Token]
---

# Moto（Latent Motion Token）

## 定义

将相邻视频帧压缩为离散**运动 token** 的框架，通过自回归方式预训练运动先验，再在动作标注轨迹上微调，作为 [[VLA]] 的运动预测中间桥梁。发表于 ICCV 2025。

## 核心架构

- **Latent Motion Tokenizer**: [[VQ-VAE]] 风格，将帧对 $(o_t, o_{t+k})$ 压缩为 $L=8$ 个离散 token，codebook $K=128$，每 token 嵌入维 $d=32$
- **Moto-GPT**: 自回归预训练，预测下一运动 token；co-fine-tuning 阶段同时预测运动 token 和机器人动作

## 核心要点

1. **两阶段训练**: 先在视频数据上预训练运动先验，再在动作标注数据上联合微调
2. **运动 token 作为桥接语言**: 在语言指令和低级动作之间插入运动 token，提升泛化性
3. **codebook 设计**: $K=128$，$L=8$，$d=32$——平衡表达力与紧凑性
4. **局限**: LAM 与 VLA 分开训练，潜在动作表示缺乏真实动作接地（LARA 的改进出发点）

## 代表工作

- 本身即代表工作（ICCV 2025）
- [[LARA]]: 采用 Moto 的 IDM+FDM 设计，通过联合训练解决 Moto 的接地问题

## 相关概念

- [[LAM]]
- [[IDM]]
- [[FDM]]
- [[VQ-VAE]]
- [[VLA]]
- [[LARA]]
