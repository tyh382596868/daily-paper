---
type: concept
aliases: [潜在交互世界模型]
---

# Latent Interactive World Model

## 定义
在紧凑的潜在空间内进行交互式世界状态预测，避免像素级重建的高计算开销。

## 数学形式
$$z_{t+1} = f_{\theta}(z_t, a_t)$$
其中 $z$ 为潜在状态，$a$ 为动作，$f$ 为潜在动力学模型。

## 核心要点
1. 世界模型在 latent space 中运行，计算效率远高于像素预测
2. 支持实时交互和闭环控制
3. 可与 VLA 策略紧密集成

## 代表工作
- [[omega-EVA]]: Envision-Verify-Act 框架的 Latent World Model
- [[Latent World Model]]: 相关概念

## 相关概念
