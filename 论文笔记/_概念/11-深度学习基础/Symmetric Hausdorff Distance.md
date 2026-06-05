---
type: concept
aliases: [HSD, 对称豪斯多夫距离]
---

# Symmetric Hausdorff Distance

## 定义

度量两个点集"最坏点对"偏差的对称版本豪斯多夫距离。

## 数学形式

$$
d_H(A, B) = \max\!\left\{ \sup_{a \in A} \inf_{b \in B} \|a-b\|_2,\; \sup_{b \in B} \inf_{a \in A} \|a-b\|_2 \right\}
$$

## 核心要点

1. 对最差点敏感、对均值不敏感——适合捕捉轨迹"飞出去"的极端偏差
2. 对噪声脆弱，常用分位数版本（partial Hausdorff）替代
3. [[Dream-exe]] 中归一化为 $[0,1]$ 后取 $1 - \tilde{d}_H$ 当作相似度

## 代表工作
- [[Dream-exe]]：用 HSD 衡量末端 / 物体轨迹形状偏差

## 相关概念
- [[Wasserstein-1 Distance]]
- [[Dynamic Time Warping]]
- [[End-Effector Trajectory]]
