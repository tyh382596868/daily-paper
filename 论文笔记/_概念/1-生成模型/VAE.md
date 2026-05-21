---
type: concept
aliases: [VAE, Variational Autoencoder, 变分自编码器]
---

# VAE

## 定义
VAE（Variational Autoencoder，变分自编码器）是一种生成模型，通过编码器把数据映射到一个概率隐空间分布、再由解码器重建数据，训练目标结合重建损失与隐空间的 KL 正则。

## 数学形式
$$
\mathcal{L}_{\mathrm{VAE}}=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)
$$

## 核心要点
1. 编码器输出隐变量分布参数（均值、方差），通过重参数化采样实现可微训练。
2. 在隐扩散模型中，VAE 提供压缩的隐空间——扩散在隐空间而非像素空间进行，大幅降低计算量。
3. 视频 VAE 还做时间维度压缩（如时间压缩比 4），把视频帧编码为隐帧序列。

## 代表工作
- [[LDM]]: 在 VAE 隐空间做扩散
- [[Wan2.1]]: 用冻结视频 VAE 编码视频帧为隐变量
- [[PROWL]]: 世界模型在冻结视频 VAE 隐空间操作，VAE 时间压缩比为 4

## 相关概念
- [[LDM]]
- [[KL 散度]]
- [[Flow Matching]]
