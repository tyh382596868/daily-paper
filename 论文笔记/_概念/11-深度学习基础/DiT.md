---
type: concept
aliases: [Diffusion Transformer, DiT]
---

# DiT

## 定义
将 Transformer 架构引入扩散模型的骨干网络，替代传统 U-Net，通过 class-conditioning 和自适应层归一化（adaLN）实现条件生成。

## 数学形式
$$\text{adaLN-Zero}: [\gamma, \beta] = \text{MLP}(c), \quad y = \gamma \cdot \text{LayerNorm}(x) + \beta$$
其中 $c$ 为条件信号（时间步 $t$ + 类别标签）。

## 核心要点
1. 用 patch embedding 将图像/latent 转化为 token 序列
2. adaLN-Zero 初始化为恒等变换，训练更稳定
3. 计算量与序列长度平方正相关，大分辨率时成本高

## 代表工作
- Peebles & Xie, 2023: DiT 原始论文（ICCV Best Paper）

## 相关概念
- [[LoRA]]
- [[EMA]]
- [[LDM]]
