---
type: concept
aliases: [RLA-WM, Residual Latent Action World Model]
---

# RLA-WM

## 定义
RLA-WM（Residual Latent Action World Model）是一种强调"预测充分性"的潜在动作世界模型：从视频帧对中学习紧凑残差潜变量作为伪动作标签，要求该潜变量能高保真地单次前向重建未来 DINO token，区别于 [[UniVLA]] 等预测充分性不足的方法。

## 核心要点
1. **残差潜在动作**：以残差形式编码当前帧到未来帧的变化，缩小表示空间
2. **预测充分性**：要求潜变量 $z$ 能从 $s_t$ 单次前向重建 $s_{t+h}$（而非模糊近似）
3. **无动作视频利用**：桥接无标注视频数据与机器人策略学习

## 数学形式
$$z = \text{Enc}(s_t, s_{t+h}), \quad \hat{s}_{t+h} = \text{Dec}(s_t, z)$$

## 代表工作
- [[RLA-WM]]: 提出残差潜在动作机制，在 Table 2 中超过 [[UniVLA]]（35.6% vs 28.7%）

## 相关概念
- [[UniVLA]]
- [[世界模型]]
- [[Diffusion Policy]]
