---
type: concept
aliases: [CVAE, Conditional VAE, 条件变分自编码器]
---

# CVAE (Conditional Variational Autoencoder)

## 定义
在 VAE 基础上加入条件变量 $c$，使编码器和解码器均以条件为输入，从而学习条件化的潜在分布 $p(z|x,c)$。

## 数学形式
$$\mathcal{L}_{CVAE} = \mathbb{E}_{q(z|x,c)}[\log p(x|z,c)] - KL(q(z|x,c) \| p(z|c))$$

## 核心要点
1. 条件 $c$ 可以是类别标签、文本、图像、地形类型等任意模态
2. 推理时给定条件 $c$，从先验 $p(z|c)$ 采样，由 decoder 生成输出
3. 比标准 VAE 生成更可控的输出，不依赖特定输入 $x$
4. 常用于运动生成、轨迹预测等结构化输出任务

## 代表工作
- [[T-GMP]]: 用 CVAE 学习地形条件的潜在运动流形，用于人形机器人步态生成

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[T-GMP]]
