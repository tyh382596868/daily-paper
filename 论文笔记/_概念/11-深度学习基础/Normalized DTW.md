---
type: concept
aliases: [NDTW, Path-normalized DTW]
---

# Normalized DTW

## 定义

把 [[Dynamic Time Warping|DTW]] 累计代价除以最优对齐路径长度，得到与序列长度无关的平均偏差。

## 数学形式

$$
\mathrm{NDTW}(A, B) = \frac{1}{|\pi^\star|}\sum_{(i,j)\in\pi^\star} \|a_i - b_j\|_2
$$

## 核心要点

1. 解决原始 DTW 随序列长度线性增长的尺度问题
2. 在导航 / 操作轨迹评测中常作为主要指标（如 VLN 中的 nDTW）
3. [[Dream-exe]] 把 NDTW 再映射到 $[0,1]$ 相似度

## 代表工作
- [[Dream-exe]]
- VLN 系列工作

## 相关概念
- [[Dynamic Time Warping]]
- [[End-Effector Trajectory]]
