---
type: concept
aliases: [Forward Dynamics Model, 前向动力学模型, FDM]
---

# FDM（Forward Dynamics Model）

## 定义

给定当前观测帧 $o_t$ 和潜在动作 $\hat{z}$，预测未来帧 $\hat{o}_{t+k}$ 的模型。是 [[LAM]] 的核心组件之一，与 [[IDM]] 配合构成完整的自监督视觉动力学学习系统。

## 数学形式

$$
\hat{o}_{t+k} = \text{FDM}(o_t, \hat{z})
$$

训练目标（重建损失）：

$$
\mathcal{L}_{recon} = \| o_{t+k} - \text{FDM}(o_t, \hat{z}) \|_2^2
$$

## 核心要点

1. **验证潜在动作的语义**: 能否从 $o_t$ 和 $\hat{z}$ 重建 $o_{t+k}$ 是 IDM 学习质量的间接验证
2. **正则化 VLA 的动作预测**: LARA 中 FDM 的前向预测作为 VLA 的约束，减少功能无效的幻觉动作
3. **推理时仅需 IDM**: FDM 主要在训练阶段使用，推理只需 IDM 提供潜在动作
4. **ViT 解码器实现**: 通常与 IDM 共用编码器，解码器负责像素/特征重建

## 代表工作

- [[LARA]]: FDM 的前向预测正则化 VLA，抑制运动学合理但任务无关的幻觉轨迹
- [[Moto]]: IDM + FDM 联合训练学习运动 token
- [[LAM]]: 通用潜在动作模型框架

## 相关概念

- [[IDM]]
- [[LAM]]
- [[VQ-VAE]]
- [[LARA]]
- [[Diffusion Policy]]
