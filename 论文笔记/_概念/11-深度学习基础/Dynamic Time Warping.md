---
type: concept
aliases: [DTW, 动态时间规整]
---

# Dynamic Time Warping

## 定义

通过动态规划在两条时序之间寻找最佳单调对齐路径，并返回最小累计点对距离。

## 数学形式

设代价 $c(i,j) = \|a_i - b_j\|$，则

$$
D(i,j) = c(i,j) + \min\big\{ D(i-1, j),\; D(i, j-1),\; D(i-1, j-1)\big\}
$$

最终 DTW 距离为 $D(|A|, |B|)$。归一化版本 [[Normalized DTW|NDTW]] = $D / |\pi^\star|$。

## 核心要点

1. 允许两序列速度不一致，对节奏差异鲁棒
2. 复杂度 $O(|A||B|)$；可用 Sakoe-Chiba 带或 FastDTW 加速
3. [[Dream-exe]] 使用 NDTW 评估生成轨迹与参考轨迹的时间对齐质量

## 代表工作
- [[Dream-exe]]
- Sakoe & Chiba (1978) 原始论文

## 相关概念
- [[Normalized DTW]]
- [[Wasserstein-1 Distance]]
- [[Symmetric Hausdorff Distance]]
