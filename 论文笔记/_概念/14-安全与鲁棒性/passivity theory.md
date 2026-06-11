---
type: concept
aliases: [无源性理论, passivity, energy tank, 能量罐]
---

# Passivity Theory

## 定义
控制理论中基于能量的稳定性框架：若一个系统消耗的能量始终不超过输入的能量，则该系统为无源（passive）系统，其与任意无源环境的互联在 Lyapunov 意义下稳定。

## 数学形式
$$\int_0^t u(\tau)^T y(\tau) d\tau \geq -\beta, \quad \forall t \geq 0$$

Energy-tank 实现：
$$\dot{E}_T = u^T y - u_d^T y_d, \quad E_T \geq 0$$

当 $E_T$ 接近耗尽时（$E_T < \epsilon$），屏蔽层拒绝可能违反无源性的控制提案。

## 核心要点
1. 无源性是系统与未知外部环境安全互联的充分条件，不依赖精确的接触模型
2. Energy-tank（能量罐）是一种离散实现：预充一定能量，控制动作消耗罐中能量，罐空则拒绝动作
3. 无源性屏蔽层可独立于上层规划器以高频运行，实现「规划慢、安全保障快」的双层架构
4. [[PaCo-VLA]] 将其用于保护 VLA 的接触操作输出，防止低频 VLA 命令引发不稳定接触

## 代表工作
- [[PaCo-VLA]]：将无源性屏蔽层与 VLA 柔顺性提案结合，用于接触丰富操作

## 相关概念
- [[admittance control]]（常与无源性屏蔽结合使用）
- [[CBF]]（Control Barrier Function，另一种安全约束方法）
- [[Impedance Control]]
