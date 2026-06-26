---
type: concept
aliases: [运动归一化预测量, motion-normalized predictor, 动态归一化幻觉指标]
---

# Motion-Normalized Predictor

## 定义

对幻觉检测信号（$u_r$、$u_f$、$u_s$）按场景运动幅度归一化的变体，消除高动态场景天然产生的高预测值干扰，使跨任务对比更公平。

## 数学形式

$$
u^{\text{norm}} = \frac{u}{m}
$$

其中运动幅度 $m$ 定义为：
$$
m = \mathrm{RMS}(\hat{z}_{t+1} - \hat{z}_t)
$$

可用数据集逐任务均值离线估计，也可在线累积计算。

## 核心要点

1. 高动态任务（如运动控制）的幻觉信号天然较大，归一化使信号与场景活跃度解耦
2. 归一化后的 $u_r^{\text{norm}}$、$u_f^{\text{norm}}$、$u_s^{\text{norm}}$ 在分类任务上 AUROC 稳定优于未归一化版本
3. 同一框架可用于不同动态幅度的任务集

## 代表工作

- [[MMBench2]]：验证归一化版本在 AUROC 上持续优于原始指标和基线（latent motion $m$、per-task 帧数）

## 相关概念

- [[Round-Trip Residual]]
- [[Flow Instability]]
- [[Inter-Seed Variance]]
- [[世界模型]]
