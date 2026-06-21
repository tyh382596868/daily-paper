---
type: concept
aliases: [Euler Integration, Euler Method, 欧拉方法, 欧拉数值积分]
---

# Euler 积分

## 定义

用于数值求解常微分方程（ODE）的最简一阶方法：以当前位置和速度场的估计值，以固定步长向前推进，从噪声分布出发到达目标分布。

## 数学形式

$$
x_{k+1} = x_k + h \cdot v(x_k, t_k)
$$

其中 $h = 1/N$ 为步长（$N$ 为总步数），$v$ 为速度场网络。

## 核心要点

1. **用于 Flow Matching 推理**: [[Flow Matching]] 训练速度场 $v_\phi$，推理时用 Euler 方法沿 ODE 积分，将标准噪声转换为目标动作分布
2. **步数与质量权衡**: 步数越少推理越快（1-step 最快），步数越多误差越小；实践中 5–10 步常达到良好平衡
3. **与 DDIM 类比**: 类似 [[DDIM]] 在扩散模型中的角色——用少步确定性采样替代多步随机采样

## 在 ω-EVA 中的使用

[[omega-EVA]] Stage 2 Flow 策略使用 **5 步 Euler 积分**在推理时生成动作块提议，兼顾速度与质量。

## 相关概念

- [[Flow Matching]]
- [[probability-flow ODE]]
- [[Rectified Flow]]
- [[11-深度学习基础]]
