---
type: concept
aliases: [Rectified Consistency Model, rCM]
---

# rCM

## 定义
结合 Consistency Model（CM，正向散度）和 Distribution Matching Distillation（DMD，反向散度）的高级扩散蒸馏框架，两者互补：CM 提供稳定初始化，DMD 提供分布对齐。

## 核心要点
1. **CM + DMD 互补**：CM（正向 KL）提供 mode-covering 的初始化，DMD（反向 KL）收紧 mode-seeking 的生成质量
2. 实现 1-2 step 高质量图像/视频生成，超越单独使用 CM 或 DMD
3. 扩展到 autoregressive 视频：[[Causal-rCM]] = 因果版 rCM，加入 Teacher-Forcing / Self-Forcing 范式
4. 连续时间版本：[[sCM]]（scaled CM）/ [[MeanFlow]] 比离散时间 CM 快 10×

## 代表工作
- [[Causal-rCM]]: autoregressive 视频扩散版本，应用于 Cosmos 3

## 相关概念
- [[Consistency Model]]
- [[DMD]]
- [[Self-Forcing]]
- [[Teacher Forcing]]
