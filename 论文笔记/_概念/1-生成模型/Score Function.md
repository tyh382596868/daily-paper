---
type: concept
aliases: [Score Function, 得分函数, 得分, Score]
---

# Score Function

## 定义
得分函数指对数概率密度关于数据的梯度 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$，它指向概率密度增大的方向，是[[扩散模型]]反向采样的核心量。

## 数学形式

$$
s_\theta(\mathbf{x}_t,t) = \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)
$$

## 核心要点
1. 扩散模型预测的噪声 $\epsilon_\theta$ 与得分函数仅差一个调度系数，二者可互相换算。
2. 得分函数对乘积分布天然友好：分布相乘 → 对数相加 → 得分相加，这正是 [[Product of Experts]] 组合的数学基础。
3. 在 [[CoME]] 中，多个记忆专家的复合得分是各专家条件得分与无条件得分按对比系数 $\alpha_k$ 的线性组合。

## 代表工作
- [[CoME]]: 推理时把各记忆专家的得分函数线性组合

## 相关概念
- [[扩散模型]]
- [[采样]]
- [[Product of Experts]]
