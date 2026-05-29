---
type: concept
aliases: [阻抗控制, Impedance Controller, 柔顺控制]
---

# Impedance Control

## 定义

Impedance Control（阻抗控制）是机器人柔顺控制的核心范式：把机械臂等效为一个**质量-弹簧-阻尼系统** $M\ddot{e} + D\dot{e} + Ke = F_{ext}$，通过指令位置/速度与实测之间的偏差产生力矩，使机械臂在与环境交互时表现出可调的"柔软度"。

## 数学形式

简化的笛卡尔阻抗律：

$$
\mathbf{u}_{cmd} = \mathbf{K}_p (\mathbf{x}_d - \mathbf{x}) + \mathbf{K}_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
$$

其中 $\mathbf{K}_p, \mathbf{K}_d$ 为可调的刚度 / 阻尼矩阵；$\mathbf{x}_d, \dot{\mathbf{x}}_d$ 来自策略输出的目标位置与目标速度。

## 核心要点

1. **柔顺操作必备**：插装、布料折叠、水杯搬运等任务需要可控柔软度；
2. **D 项依赖速度质量**：传统离散动作策略的 $\dot{\mathbf{x}}_d$ 用有限差分估计，噪声大；
3. **NIAF 的关键收益**：[[NIAF]] 通过 [[SIREN]] 解析求导直接给出 $\dot\Phi$，让 D 项无噪声；
4. **与 Admittance Control 对偶**：阻抗 = 位置入 → 力出；导纳 = 力入 → 位置出。

## 代表工作

- Hogan, 1985：经典阻抗控制公式化
- [[NIAF]]：用解析速度替换有限差分速度，水杯搬运面波动 ↓60%

## 相关概念

- [[Jerk]]
- [[Neural Implicit Action Field]]
- [[MPC]]
- [[PID]]
