---
type: concept
aliases: [Kabsch Algorithm, Kabsch 算法, Procrustes Rotation]
---

# Kabsch Alignment

## 定义

在最小二乘意义下求解两组 3D 点对之间最优旋转矩阵的解析算法，基于 [[SVD]]。

## 数学形式

给定去质心的对应点集 $\{p_i\}, \{q_i\}$：

$$
H = \sum_i p_i q_i^\top,\quad H = U\Sigma V^\top
$$
$$
R^\star = V\, \mathrm{diag}(1, 1, \det(V U^\top))\, U^\top
$$

中间的 $\det$ 项用于消除反射，确保 $R^\star \in \mathrm{SO}(3)$。

## 核心要点

1. 解析解，无需迭代，复杂度 $O(N + \text{SVD}(3\times3))$
2. 输入要求两组点对一一对应且已去质心
3. 鲁棒变体：加权 Kabsch、RANSAC + Kabsch

## 代表工作
- [[Dream-exe]]：用 Kabsch 从 3D 跟踪点恢复末端姿态
- [[Pi3X]]：3D 重建中对齐多视图坐标系

## 相关概念
- [[SVD]]
- [[Rotation Matrix]]
- [[ICP]]
