---
type: concept
aliases: [UNet, U形网络, 编码器-解码器网络]
---

# U-Net

## 定义

U-Net 是一种对称的编码器-解码器卷积神经网络架构，以跳跃连接（skip connections）将编码器各层特征直接传递到对应解码器层，从而在下采样-上采样过程中保留细节信息。

## 数学形式

**跳跃连接**:

$$
h_l^{\text{dec}} = \text{UpConv}(h_{l+1}^{\text{dec}}) \oplus h_l^{\text{enc}}
$$

其中 $\oplus$ 为特征拼接，$h_l^{\text{enc}}$ 为编码器第 $l$ 层特征，$h_l^{\text{dec}}$ 为解码器对应层输出。

## 核心要点

1. **对称结构**: 编码器（下采样）与解码器（上采样）路径对称，中间为瓶颈层
2. **跳跃连接**: 直接将编码器层特征拼接到解码器，缓解梯度消失，保留局部细节
3. **FiLM 条件化**: 在扩散模型中，常用 FiLM（Feature-wise Linear Modulation）将时间步和条件信息注入 U-Net 各层
4. **广泛应用**: 图像分割（原始用途）、图像生成、扩散模型去噪网络

## 代表工作

- [[LAD]]: 在 mimic 手 + Franka 夹爪实验中使用 U-Net-based [[Diffusion Policy]]（FiLM 条件化）
- [[Diffusion Policy]]: 提出用 U-Net 作为扩散策略去噪网络

## 相关概念

- [[Transformer]]
- [[Diffusion Policy]]
- [[DDPM]]
