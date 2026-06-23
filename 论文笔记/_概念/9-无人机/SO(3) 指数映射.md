---
type: concept
aliases: [SO3 指数映射, SO(3) Exponential Map, 旋转指数映射, Lie群指数映射]
---

# SO(3) 指数映射

## 定义

SO(3) 指数映射是将李代数 $\mathfrak{so}(3)$（三维反对称矩阵空间）映射到李群 $SO(3)$（三维旋转矩阵群）的映射，用于以几何一致的方式积分角速度更新姿态，保证旋转矩阵始终满足正交约束 $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$，$\det(\mathbf{R})=1$。

## 数学形式

$$
\mathbf{R}_{t+1} = \mathbf{R}_t \exp([\boldsymbol{\omega}_t]_\times \Delta t)
$$

其中 Rodrigues 公式给出：

$$
\exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2, \quad \theta = \|\boldsymbol{\omega}\|
$$

## 核心要点

1. **几何一致性**: 直接欧拉积分 $\mathbf{R}_{t+1} = \mathbf{R}_t + [\boldsymbol{\omega}_t]_\times \mathbf{R}_t \Delta t$ 会破坏正交性，长时积分后旋转矩阵漂移；指数映射天然保持 $SO(3)$ 结构
2. **反对称矩阵 $[\boldsymbol{\omega}]_\times$**: 将角速度向量 $\boldsymbol{\omega}\in\mathbb{R}^3$ 映射为 $3\times3$ 反对称矩阵
3. **对偶 Log 映射**: $\mathrm{Log}_{SO(3)}(\mathbf{R})$ 可从旋转矩阵恢复旋转向量，用于误差计算

## 代表工作

- [[SkyJEPA]]: 在 Physics-Inspired Prober 的状态积分中使用 SO(3) 指数映射更新旋转矩阵，保持姿态预测的几何合理性

## 相关概念

- [[物理启发式探针]]: SkyJEPA 中使用 SO(3) 指数映射的模块
- [[MPC]]: 在旋转感知 MPC 中姿态误差通常也在 SO(3) 上计算
