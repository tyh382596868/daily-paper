---
type: concept
aliases: [流匹配, Conditional Flow Matching, CFM]
---

# Flow Matching

## 定义

一种基于连续正则化流的生成模型训练范式，通过直接回归从噪声到数据的速度场（velocity field）来训练生成模型，比 DDPM 更简洁高效。

## 数学形式

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t, q(x_1), p(x_0)} \| v_\theta(x_t, t) - u_t(x_t | x_1) \|^2
$$

其中 $u_t(x_t | x_1) = x_1 - x_0$ 为从噪声 $x_0$ 到数据 $x_1$ 的条件速度场，$x_t = (1-t)x_0 + t x_1$ 为线性插值路径。

## 核心要点

1. 目标是学习一个向量场 $v_\theta$，将噪声分布通过 ODE 积分变换为数据分布
2. 线性路径（Rectified Flow）比 DDPM 的马尔可夫链更直接，推理步数更少
3. 时间步权重 $\omega(t)$ 可灵活调节不同时间步的损失贡献
4. 已成为大规模视频/图像生成（如 Wan2.2、SD3）的主流训练目标

## 代表工作

- [[Wan2.2]]: 基于 Flow Matching 训练的视频生成模型
- [[EA-WM]]: 使用 Flow Matching 目标联合训练视频和 KVAF 预测

## 相关概念

- [[视频扩散模型]]
- [[扩散变换器]]
- [[LDM]]
