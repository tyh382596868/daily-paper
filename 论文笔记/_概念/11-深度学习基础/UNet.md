---
type: concept
aliases: [UNet, U-Net, 编解码网络]
---

# UNet

## 定义

UNet 是 Ronneberger et al. (2015) 提出的编解码对称架构，通过跳跃连接将编码器特征传递给解码器，广泛应用于图像分割和扩散模型的去噪网络。

## 数学形式

$$
\hat{x} = \text{Decoder}(z, \text{skip\_connections}(\text{Encoder}(x)))
$$

## 核心要点

1. 对称的编解码结构，编码器逐步下采样，解码器逐步上采样
2. 跳跃连接（skip connections）将高分辨率特征直接传入解码器，保留细节信息
3. 在扩散模型中作为去噪网络，输入加噪图像和时间步，输出预测噪声
4. 已被 [[DiT]] (Diffusion Transformer) 替代成为新一代扩散骨干

## 代表工作

- [[EA-WM]]: EA-WM 基于 Transformer 架构而非 UNet，作为对比背景提及

## 相关概念

- [[视频扩散模型]]
- [[DiT]]
