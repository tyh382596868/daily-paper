---
type: concept
aliases: [Joint Embedding Predictive Architecture, 联合嵌入预测架构]
---

# JEPA

## 定义
LeCun 提出的自监督学习框架，在潜空间中预测表示而非像素，通过最小化 latent embedding 之间的预测误差来学习世界模型。

## 数学形式
$$\mathcal{L} = \| s_\phi(x_{t+k}) - p_\psi(s_\phi(x_t), a_{t:t+k}) \|^2$$
其中 $s_\phi$ 为 encoder，$p_\psi$ 为预测器，避免在像素空间重建。

## 核心要点
1. 核心思想：预测 latent 表示，而非重建像素（避免 collapse）
2. 通常用 EMA（指数移动平均）更新目标 encoder 防止 collapse
3. I-JEPA（图像）、V-JEPA（视频）是已发布的实现

## 代表工作
- Assran et al., 2023: [[I-JEPA]]
- Bardes et al., 2024: [[V-JEPA]]
- [[LeWM]]: 基于 [[SIGReg]] 的稳定端到端 JEPA 世界模型
- [[PLDM]]: 此前唯一的端到端像素 JEPA 基线，需要 6+ 损失项
- [[SkyJEPA]]: 将 JEPA 应用于四旋翼动力学建模，配合物理探针实现零样本 Sim-to-Real 控制

## 相关概念
- [[I-JEPA]]
- [[V-JEPA]]
- [[SIGReg]]
- [[VICReg]]
- [[表征坍塌]]
- [[DreamerV3]]
- [[DINO]]
- [[EMA]]
