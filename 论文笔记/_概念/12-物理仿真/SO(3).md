---
type: concept
aliases: [SO3, 特殊正交群, 旋转群, Special Orthogonal Group]
---

# SO(3)

## 定义

SO(3) 是三维空间中所有旋转变换的集合，构成一个李群（Lie Group），其元素为行列式为 +1 的 3×3 正交矩阵 $\mathbf{R}$。

## 数学形式

$$
\mathrm{SO}(3) = \{\mathbf{R} \in \mathbb{R}^{3\times 3} \mid \mathbf{R}^\top \mathbf{R} = \mathbf{I},\ \det(\mathbf{R}) = +1\}
$$

**指数映射**（李代数 $\mathfrak{so}(3)$ → SO(3)）：

$$
\mathbf{R} = \exp([\boldsymbol{\omega}]_\times)
$$

其中 $[\boldsymbol{\omega}]_\times$ 是角速度向量 $\boldsymbol{\omega} \in \mathbb{R}^3$ 对应的反对称矩阵（skew-symmetric matrix）。

**离散积分**（旋转矩阵时序传播）：

$$
\mathbf{R}_{t+1} = \mathbf{R}_t \exp([\boldsymbol{\omega}_t]_\times \Delta t)
$$

## 核心要点

1. SO(3) 是一个 3 维紧致李群，不是欧几里得空间，不能用普通加法更新
2. 对应的李代数 $\mathfrak{so}(3)$ 是所有 3×3 反对称矩阵的空间
3. 指数映射将李代数元素（角速度）映射到群元素（旋转矩阵），确保积分后旋转矩阵仍满足正交约束
4. 相比欧拉角（3 个标量）避免万向锁（Gimbal Lock），相比四元数无需显式归一化

## 代表工作

- [[SkyJEPA]]: 使用 SO(3) 指数映射在物理探针中传播旋转矩阵，保持姿态估计的几何一致性

## 相关概念

- [[MPPI]]
- [[World Model]]
