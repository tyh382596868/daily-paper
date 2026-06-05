---
type: concept
aliases: [斯皮尔曼相关系数, Rank Correlation]
---

# Spearman Correlation

## 定义

衡量两组变量秩次单调相关性的非参数统计量。

## 数学形式

$$
\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)},\quad d_i = \mathrm{rank}(x_i) - \mathrm{rank}(y_i)
$$

## 核心要点

1. 不要求线性 / 正态，对单调非线性关系仍有效
2. 范围 $[-1, 1]$；$|\rho|$ 接近 1 表明强单调相关
3. [[Dream-exe]] 用其量化"plausibility vs 任务成功率"——结果 $r=-0.03$，证伪视觉好=世界模型好

## 相关概念
- [[Pearson Correlation]]
- [[Dream.exe Benchmark]]
