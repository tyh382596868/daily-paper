---
type: concept
aliases: [旋转矩阵, SO(3) Matrix]
---

# Rotation Matrix

## 定义

满足 $R^\top R = I,\; \det(R)=1$ 的 $3\times 3$ 实矩阵，构成李群 $\mathrm{SO}(3)$。

## 核心要点

1. 表征 3D 旋转的最直接形式，避免万向锁
2. 常通过 [[SVD]] 或四元数 / 轴角等参数化求解
3. 在末端姿态估计中常用 [[Kabsch Alignment]] 求最优旋转

## 相关概念
- [[Kabsch Alignment]]
- [[SVD]]
- [[Quaternion]]
- [[End-Effector Trajectory]]
