---
type: concept
aliases: [ICP, Iterative Closest Point, 迭代最近点]
---

# ICP (Iterative Closest Point)

## 定义

经典点云配准算法，给定两组 3D 点集 $P, Q$，迭代求解使 $P$ 经过刚体变换 $T \in SE(3)$ 后与 $Q$ 最近邻匹配距离平方和最小的 $T$。

## 数学形式

$$
T^* = \arg\min_{T \in SE(3)} \sum_{p \in P} \| T(p) - \text{NN}(T(p), Q) \|^2
$$

迭代步骤：
1. **匹配**：为每个 $p \in P$ 在 $Q$ 中找最近邻
2. **求解**：在匹配下解析求最优 $T$（用 SVD）
3. **迭代**：用新 $T$ 重新匹配，直至收敛

## 核心要点

1. **刚体变换**：求解 $SE(3)$（旋转 + 平移）
2. **闭式 SVD 子步**：每步求解有闭式解
3. **依赖初始化**：易陷入局部最优
4. **变体众多**：point-to-plane、Generalized ICP、Color ICP 等

## 代表工作

- 原始 ICP (Besl & McKay, 1992)
- [[GAF]] 把 ICP 应用在夹爪相关 3D 高斯位移场上，抽取夹爪刚体变换作为初始动作

## 相关概念

- [[Structure-from-Motion]]
- [[Gaussian Action Field]]
- [[3D Gaussian Splatting]]
