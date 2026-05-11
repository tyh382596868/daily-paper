---
type: concept
aliases: [Video-JEPA, V-JEPA, V-JEPA 2]
---

# V-JEPA

## 定义

Video-JEPA (Bardes et al., 2024) 把 [[I-JEPA]] 推广到视频：从视频片段中遮挡 spatio-temporal block，让 predictor 在 latent 空间预测被遮挡区域的 target encoder 输出。沿用 EMA target encoder + stop-gradient 防止[[表征坍塌]]。V-JEPA 2 在大规模视频数据上预训练，是视频自监督的代表方法。

## 数学形式

与 [[I-JEPA]] 相同，但 $\mathbf{x}$ 是视频片段，遮挡区域跨越时间维：

$$
\mathcal{L} = \sum_m \|\text{pred}_\phi(\text{enc}_\theta(\mathbf{x}_{\text{ctx}}), m) - \text{sg}(\text{enc}_{\bar\theta}(\mathbf{x}_{\text{tgt}}^{(m)}))\|^2
$$

## 核心要点

1. **视频自监督代表**：在 Kinetics、SSv2 等动作识别 benchmark 上达到 SOTA
2. **无需文本配对**：相比 InternVideo / VideoCLIP 仅靠视频本身
3. **作为机器人 encoder**：被用于 V-JEPA2 等动作世界模型（如 LeJEPA）

## 代表工作

- Bardes et al., 2024: V-JEPA 原始论文
- V-JEPA 2 (2025): 大规模视频预训练版
- [[LeWM]]: 在像素级 JEPA 上用 SIGReg 替代 EMA + stop-gradient

## 相关概念

- [[JEPA]]
- [[I-JEPA]]
- [[EMA]]
- [[世界模型]]
