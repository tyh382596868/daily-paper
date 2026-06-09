---
type: concept
aliases: [HMR2, HMR2.0, Human Mesh Recovery 2.0]
---

# HMR2 (Human Mesh Recovery 2.0)

## 定义

基于 ViT 的单目 3D 人体网格恢复模型，输入单张 RGB 图像，输出 SMPL 人体参数（形状 $\beta$、姿态 $\theta$），是当前单帧 3D 人体姿态估计的主流基线之一。

## 数学形式

$$
(\theta, \beta) = \text{HMR2}(I) \quad \Rightarrow \quad M_{SMPL} = \text{SMPL}(\theta, \beta)
$$

## 核心要点

1. 使用 ViT-H 主干，相比 ResNet-based HMR 系列精度大幅提升
2. 在 4DHumans 系列框架中配合时序跟踪使用
3. 输出 SMPL 参数，可进一步升级到 SMPL-X（含手部和面部）
4. 弱点：单帧估计在遮挡/自遮挡时存在深度歧义

## 代表工作

- [[GRAIL]]: GEM-SMPL 中使用 HMR2 提供逐帧 SMPL-X body 估计，配合 ViTPose 和时序方法联合优化

## 相关概念

- [[SMPL]]
- [[ViTPose]]
- [[MPJPE]]
- [[GENMO]]
