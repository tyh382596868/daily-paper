---
type: concept
aliases: [因果强迫]
---

# Causal Forcing

## 定义

把双向视频扩散模型蒸馏成"相机/动作可控、因果自回归、少步采样"的世界模型的**三阶段训练范式**：Stage 1 因果 AR 教师强迫；Stage 2 [[Causal ODE Initialization|Causal ODE]] 离线少步初始化；Stage 3 [[Asymmetric DMD]] 与双向 teacher 做分布对齐。

## 数学形式

整体目标是把双向多步教师 $\epsilon_\phi$ 蒸馏成因果少步学生 $G_\theta$，三阶段串行优化，Stage 2 的关键损失见 [[Causal ODE Initialization]]。

## 核心要点

1. **离线 ODE 数据驱动**：Stage 2 用 teacher 跑 ODE 轨迹做监督，干净但存储贵。
2. **因果性 + 少步**双重压缩，bidirectional → AR → few-step 各贡献 ~10× 加速。
3. 与 [[Causal Forcing++]] 是互补的两条蒸馏路径（offline vs online）。

## 代表工作

- [[minWM]]: 同时实现了 Causal Forcing 与 Causal Forcing++，作为 pipeline 的两条 Stage 2 可选支线。

## 相关概念

- [[Causal Forcing++]]
- [[Asymmetric DMD]]
- [[Diffusion Forcing]]
- [[Teacher Forcing]]
