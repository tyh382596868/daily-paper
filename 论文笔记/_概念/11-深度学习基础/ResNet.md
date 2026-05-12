---
type: concept
aliases: [ResNet, 残差网络, ResNet-18, ResNet-50]
---

# ResNet

## 定义

ResNet（Residual Network, He et al., 2016）是引入**残差连接（skip connection）**的卷积神经网络架构：每个残差块学习 $\mathcal{F}(x)+x$ 而非直接学 $\mathcal{H}(x)$，缓解深层网络的梯度消失与退化问题，使训练上百层网络成为可能。ResNet-18/34/50 是机器人 BC 策略常用的轻量视觉 backbone。

## 数学形式

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

其中 $\mathcal{F}$ 是若干卷积层组成的残差映射，$+x$ 是恒等捷径连接。

## 核心要点

1. **残差学习**：让网络学"相对于输入的修正"，恒等映射易于优化，深度可大幅增加。
2. **常见变体**：ResNet-18/34（BasicBlock）、ResNet-50/101/152（Bottleneck）。
3. **机器人中的用途**：作为 BC 策略的图像编码器（如 [[Diffusion Policy]]、[[RLA-WM]] 中的 BC-ResNet-18 基线）；轻量、易微调，可挂 [[LoRA]] 适配器做 RL。
4. **与 Transformer/ViT 对比**：参数少、推理快，适合数据量有限的具身任务。

## 代表工作

- He et al., 2016: ResNet 原始论文
- 作为策略 backbone 出现在 [[RLA-WM]]（BC-ResNet、WMRL 的 LoRA 适配对象）

## 相关概念

- [[ViT]]
- [[Transformer]]
- [[LoRA]]
- [[Diffusion Policy]]
