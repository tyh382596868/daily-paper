---
type: concept
aliases: [EEF Trajectory, 末端轨迹, TCP Trajectory]
---

# End-Effector Trajectory

## 定义

机器人末端执行器（如夹爪 TCP）随时间变化的位置 + 姿态 + 夹爪状态序列，是大部分操作策略的输出形式。

## 数学形式

$$
\tau = \{(p_t, R_t, g_t)\}_{t=1}^T,\quad p_t \in \mathbb{R}^3,\; R_t \in \mathrm{SO}(3),\; g_t \in \{0,1\}
$$

## 核心要点

1. 是 [[VLA]] / [[Diffusion Policy]] 等策略最常见的输出空间
2. 评测时常用 HSD、Wasserstein、NDTW 等度量
3. 视觉中心与 TCP 中心的偏移需要标定（[[Dream-exe]] 用初始帧 calibration）

## 代表工作
- [[Dream-exe]]
- [[Diffusion Policy]]
- [[ACT (Action Chunking Transformer)]]

## 相关概念
- [[Action Chunking]]
- [[Video-to-Trajectory]]
- [[Symmetric Hausdorff Distance]]
- [[Normalized DTW]]
