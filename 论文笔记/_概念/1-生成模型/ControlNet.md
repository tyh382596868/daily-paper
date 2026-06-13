---
type: concept
aliases: [ControlNet]
---

# ControlNet

## 定义
Zhang et al. 2023 提出的扩散模型条件控制框架，通过冻结原始 U-Net 参数并添加可训练的 zero-convolution 控制分支，实现对生成内容的精确条件控制（边缘图、深度图、关键点等）。

## 数学形式
$$y = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c); \Theta_{z2})$$

其中 $\mathcal{Z}$ 为 zero-convolution（初始化为零的 1×1 卷积），$c$ 为条件输入。

## 核心要点
1. 冻结预训练模型，只训练控制分支（防止灾难遗忘）
2. zero-convolution 保证训练初期不破坏原始模型输出
3. 支持多种控制信号：Canny 边缘、HED、深度图、法线图、骨骼关键点

## 代表工作
- [[DDPM]] / [[DDIM]] — 底层扩散模型

## 相关概念
- [[Diffusion Model]]
- [[Classifier-Free Guidance]]
