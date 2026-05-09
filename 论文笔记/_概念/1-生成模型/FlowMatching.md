---
type: concept
aliases: [Flow Matching, 流匹配, 条件流匹配, Conditional Flow Matching]
---

# Flow Matching

## 定义
Flow Matching 是一种生成模型训练范式，通过回归从噪声分布到数据分布的连续向量场（速度场），实现高效的样本生成，无需 SDE/ODE 求解器的迭代扩散过程。

## 数学形式

训练目标（最小化速度场预测误差）：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\tau,\epsilon}\left\| \mathbf{u}_\theta(\mathbf{x}^\tau, \tau, \mathbf{c}) - (\mathbf{x} - \epsilon) \right\|^2_2
$$

含噪样本构造（线性插值）：

$$
\mathbf{x}^\tau = \tau \mathbf{x} + (1 - \tau)\epsilon, \quad \tau \sim \mathcal{U}(0,1),\; \epsilon \sim \mathcal{N}(0,I)
$$

推理采样（Euler 积分）：

$$
\mathbf{x}^{\tau_{i+1}} = \mathbf{x}^{\tau_i} + (\tau_{i+1} - \tau_i)\, \mathbf{u}_\theta(\mathbf{x}^{\tau_i}, \tau_i, \mathbf{c})
$$

## 核心要点

1. **线性插值路径**: 不同于扩散模型的非线性噪声调度，Flow Matching 使用数据与噪声的线性插值，路径更直，少步推理质量高
2. **速度场回归**: 直接监督速度场 $\mathbf{u}_\theta$（=数据-噪声），比 score matching 更直接、更稳定
3. **推理高效**: 少量 Euler 步（通常 10 步内）即可生成高质量样本
4. **条件生成**: 通过条件输入 $\mathbf{c}$（语言、图像、状态）控制生成方向，适合策略学习

## 代表工作

- [[pi0]]: 首批将 Flow Matching 引入机器人策略的 VLA 模型
- [[RLDX-1]]: 采用 Flow Matching 生成动作块，并扩展到物理信号预测

## 相关概念

- [[扩散模型|Diffusion Model]]
- [[Action Chunking]]
- [[VLA]]
