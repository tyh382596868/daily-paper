---
type: concept
aliases: [高斯过程, GP, Gaussian Process Regression]
---

# Gaussian Process

## 定义

高斯过程（Gaussian Process, GP）是定义在函数空间上的概率分布——任意有限个点处的函数值服从联合高斯分布。GP 由均值函数 $m(x)$ 和核函数（协方差函数）$k(x, x')$ 完全刻画，常用于非参数贝叶斯回归和轨迹生成。

## 数学形式

$$
f(x) \sim \mathcal{GP}(m(x),\ k(x, x'))
$$

对于 $n$ 个观测点，联合分布为：

$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})
$$

其中 $K_{ij} = k(x_i, x_j)$。

**轨迹生成（周期核叠加）**：

$$
p_j(t) \sim \mathcal{GP}(0,\ k_j(t, t')), \quad j \in \{x, y, z\}
$$

核函数为不同长度尺度和周期的周期核之和，生成平滑且多样的参考轨迹。

## 核心要点

1. 无需指定参数化函数形式，通过核函数直接编码先验（平滑性、周期性等）
2. 提供预测不确定度估计，适合数据稀疏场景
3. 标准 GP 计算复杂度 $O(n^3)$，大规模场景需稀疏近似
4. 在机器人学中常用于生成覆盖多样动态的参考轨迹，配合域随机化扩充训练数据

## 代表工作

- [[SkyJEPA]]: 用周期核叠加的 GP 生成 20,000 条 10 s 参考轨迹（XYZ 独立采样），确保训练数据的轨迹多样性

## 相关概念

- [[Domain Randomization]]
- [[World Model]]
