---
type: concept
aliases: [Neural External Torque Estimation, 神经外部力矩估计]
---

# NEXT（Neural External Torque Estimation）

## 定义
NEXT 是 FACTR 2 提出的数据驱动外部关节力矩估计器，无需专用力/力矩传感器，仅利用标准关节编码器数据估计接触力，适配商用机械臂。

## 数学形式
外部力矩估计：
$$\hat{\boldsymbol{\tau}}_{\text{ext}} = f_\theta(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}, \boldsymbol{\tau}_{\text{cmd}})$$

其中 $\mathbf{q}$ 为关节角度，$\dot{\mathbf{q}}$ 为关节速度，$\boldsymbol{\tau}_{\text{cmd}}$ 为指令力矩，$f_\theta$ 为神经网络。

训练数据：在机械臂已知位置施加标定外力，记录关节编码器读数：
$$\mathcal{D} = \{(\mathbf{q}_t, \dot{\mathbf{q}}_t, \boldsymbol{\tau}^{\text{ref}}_{\text{ext},t})\}_{t=1}^T$$

## 核心要点
1. 训练仅需 ~1 小时标定数据，无需额外硬件
2. 估计值直接作为策略输入，提升 contact-rich 任务成功率
3. 相比专用 F/T 传感器，成本接近零
4. 主要误差来源：关节摩擦建模误差、传感器噪声

## 代表工作
- [[FACTR2]]: NEXT 提出论文，商用机械臂力感知

## 相关概念
- [[Impedance Control]]
- [[ACT (Action Chunking Transformer)]]
- [[admittance control]]
