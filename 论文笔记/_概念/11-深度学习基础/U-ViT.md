---
type: concept
aliases: [U-ViT, U-shaped ViT, UViT]
---

# U-ViT

## 定义
一种把 U-Net 的"长跳连接(long skip connection)"思想引入 Vision Transformer 的扩散模型骨干：所有输入（带噪图像 patch、时间步、条件）都当作 token，在浅层与深层之间加跳连接，兼具 Transformer 的可扩展性与 U-Net 的多尺度特性。

## 核心要点
1. 纯 Transformer 实现，无卷积下采样/上采样，靠 token 化 + skip connection 完成"U 形"信息流。
2. 是 [[DiT]] 之外另一条主流扩散骨干路线，常用于图像/3D 扩散。
3. 在 3D-Belief 中作为去噪网络主干，上接几何头与语义头。

## 代表工作
- [[3D-Belief]]: 用共享 U-ViT 主干 + MVS 风格 3DGS 几何头 + CLIP 蒸馏语义头

## 相关概念
- [[ViT]]
- [[DiT]]
- [[UNet]]
- [[扩散模型]]
