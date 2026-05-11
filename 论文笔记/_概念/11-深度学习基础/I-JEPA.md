---
type: concept
aliases: [Image-JEPA, I-JEPA]
---

# I-JEPA

## 定义

Image-JEPA (Assran et al., CVPR 2023) 是 [[JEPA]] 在图像上的代表实现：从一张图像中遮挡若干 target block，让 predictor 在 latent 空间预测被遮挡区域的 target encoder 输出。target encoder 用 [[EMA]] 维护，predictor 通过 stop-gradient 防止[[表征坍塌]]。

## 数学形式

$$
\mathcal{L} = \sum_{m} \|\text{pred}_\phi\big(\text{enc}_\theta(\mathbf{x}_{\text{ctx}}),\, m\big) - \text{sg}\big(\text{enc}_{\bar\theta}(\mathbf{x}_{\text{tgt}}^{(m)})\big)\|^2
$$

其中 $\bar\theta$ 是 EMA 更新的 target 参数，$\text{sg}$ 是 stop-gradient。

## 核心要点

1. **像素级 → latent 级**：相比 MAE 在像素空间重建，I-JEPA 在 latent 空间重建，避免学习无关纹理
2. **依赖 EMA + stop-gradient**：缺乏理论保障的反坍塌手段
3. **强 zero-shot / linear-probe 性能**：在 ImageNet 上接近 DINOv2

## 代表工作

- Assran et al., CVPR 2023: 原始论文
- [[V-JEPA]]: 视频版本扩展
- [[LeWM]]: 摆脱了 I-JEPA 对 EMA + stop-gradient 的依赖

## 相关概念

- [[JEPA]]
- [[V-JEPA]]
- [[EMA]]
- [[表征坍塌]]
