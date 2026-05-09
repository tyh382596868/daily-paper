---
type: concept
aliases: [Model Predictive Control, 模型预测控制]
---

# MPC

## 定义
一种基于模型的在线优化控制方法，在每个时间步滚动优化未来有限时域内的控制序列，并只执行第一步控制输入。

## 数学形式
$$\min_{u_{0:H-1}} \sum_{t=0}^{H-1} l(x_t, u_t) + l_f(x_H)$$
$$\text{s.t.} \quad x_{t+1} = f(x_t, u_t), \quad x_t \in \mathcal{X}, \quad u_t \in \mathcal{U}$$

## 核心要点
1. Receding horizon：每步重新优化，适应模型误差
2. 需要精确的动力学模型 $f$，在世界模型中替换为学习得到的 latent dynamics
3. 常与 [[CEM]] 结合做随机采样优化

## 相关概念
- [[CEM]]
- [[DreamerV3]]
- [[JEPA]]
