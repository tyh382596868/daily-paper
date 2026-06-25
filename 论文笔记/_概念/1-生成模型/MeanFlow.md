---
type: concept
aliases: [MeanFlow, Mean Flow Matching]
---

# MeanFlow

## 定义
连续时间扩散蒸馏方法，用平均速度场（mean velocity field）参数化 flow，比离散时间 Consistency Model 收敛更快、训练更稳定，属于 [[sCM]] 系列的变体。

## 核心要点
1. 参数化 $v_\theta(x_t, t)$ 为 $x_0$ 到 $x_t$ 的平均速度，而非瞬时速度
2. 避免离散时间 CM 中的序列长度超参数选择问题
3. 与 [[sCM]] 的区别：MeanFlow 用平均速度场，sCM 用缩放一致性条件
4. 在 [[Causal-rCM]] 中：MeanFlow 作为 Teacher-Forcing 的连续时间 CM 变体，比 dCM 快 10×

## 数学形式
$$v_\theta(x_t, t) = \frac{x_t - x_0}{t}$$（理想情况下的平均速度场）

## 代表工作
- [[Causal-rCM]]: 用 MeanFlow 实现 autoregressive 视频扩散的连续时间蒸馏

## 相关概念
- [[Consistency Model]]
- [[rCM]]
- [[DMD]]
