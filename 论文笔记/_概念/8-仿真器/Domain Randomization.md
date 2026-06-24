---
type: concept
aliases: [域随机化, DR, 参数随机化]
---

# Domain Randomization

## 定义

Domain Randomization（域随机化）是一种 Sim-to-Real 迁移技术：在训练时系统性地随机化仿真环境的物理参数（质量、摩擦、惯量等），使模型学到对参数扰动鲁棒的策略，从而在不同真实硬件平台上零样本泛化。

## 数学形式

从参数分布中采样随机化域：

$$
\boldsymbol{\eta}_r \sim \mathcal{U}(\bar{\boldsymbol{\eta}} - \Delta\boldsymbol{\eta},\ \bar{\boldsymbol{\eta}} + \Delta\boldsymbol{\eta})
$$

其中 $\bar{\boldsymbol{\eta}}$ 为标称参数，$\Delta\boldsymbol{\eta}$ 为随机化范围。常见随机化参数集合：

$$
\boldsymbol{\eta} = \{m, D, J, \alpha, k_f, k_\tau, l\}
$$

（质量、阻力、转动惯量、电机时间常数、推力系数、力矩系数、臂长）

## 核心要点

1. 不需要真实数据，完全在仿真中训练，降低数据采集成本
2. 随机化范围需覆盖真实世界的参数变化范围（不能过窄也不能过宽）
3. 是 Sim-to-Real 的核心方法之一，与 [[Adaptive Control]] 和在线微调互补
4. 通常配合 Gaussian Process 参考轨迹生成多样化动作分布，避免策略过拟合

## 代表工作

- [[SkyJEPA]]: 500 个随机化域 × 20,000 条轨迹，实现四旋翼飞行器零样本 Sim-to-Real 部署

## 相关概念

- [[World Model]]
- [[MPPI]]
