---
type: concept
aliases: [DPT, Dense Prediction Transformer]
---

# DPT (Dense Prediction Transformer)

## 定义

由 Ranftl et al. (Intel ISL) 提出的稠密预测 transformer decoder 架构，将 ViT 不同层的 token 重组为多尺度特征图，通过 fusion block 渐进上采样实现像素级输出（深度、分割等）。是 [[Depth Anything V2|Depth Anything]] 等深度估计模型的标准解码 head。

## 核心要点

1. **Reassemble 操作**: 把 ViT 每层 $H/p \times W/p$ token 重排为不同分辨率特征图（$\frac{H}{32}, \frac{H}{16}, \frac{H}{8}, \frac{H}{4}$）
2. **Fusion block**: 自顶向下 + 残差融合，对应 [[U-Net]] / FPN 风格的渐进解码
3. **稠密输出**: 最终上采样到原图分辨率，回归连续值（深度）或分类（分割）
4. **与 CNN 对比**: 保留 ViT 的全局感受野，比卷积 backbone 的稠密预测精度更高

## 代表工作

- [[Depth Anything V2]]: DPT 解码 head + DINOv2 编码器
- [[GaussianDream]]: 借鉴 DPT 风格的特征 fusion 设计上采样 backbone $B_G$

## 相关概念

- [[ViT]]
- [[DINOv2]]
- [[Depth Anything V2]]
