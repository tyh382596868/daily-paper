---
type: concept
aliases: [BatchNorm, Batch Normalization, 批归一化]
---

# BatchNorm

## 定义

Batch Normalization 是 Ioffe & Szegedy (2015) 提出的归一化层：在 mini-batch 维度上对每个特征通道做零均值单位方差归一化，再用可学习的 $\gamma,\beta$ 做仿射重投影。可显著加速训练并稳定深度网络。

## 数学形式

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu_B, \sigma_B^2$ 是 mini-batch 内的均值和方差。

## 核心要点

1. **训练 vs 推理**：训练时用 batch 统计量，推理时用 EMA 估计的总体统计量
2. **替代方案**：[[LayerNorm]] / GroupNorm / InstanceNorm 在小 batch 或序列任务下更稳
3. **作为 projector 模块**：在 SimCLR、BYOL、SIGReg 等自监督方法中常用作投影层归一化

## 代表工作

- Ioffe & Szegedy, 2015: 原始论文
- [[LeWM]]: encoder/predictor 的 projector 用 1 层 BatchNorm MLP

## 相关概念

- [[Transformer]]
- LayerNorm
