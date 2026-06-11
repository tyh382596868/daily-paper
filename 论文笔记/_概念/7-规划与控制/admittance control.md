---
type: concept
aliases: [导纳控制, admittance controller]
---

# Admittance Control

## 定义
一种力/位置混合控制策略，将外部接触力映射为期望的位置/速度修正量，使机器人末端执行器呈现类弹簧-阻尼的柔顺行为。

## 数学形式
$$M_d \ddot{x}_e + B_d \dot{x}_e + K_d x_e = F_{\text{ext}}$$

其中 $M_d$、$B_d$、$K_d$ 分别为期望质量、阻尼、刚度矩阵，$F_{\text{ext}}$ 为外部接触力，$x_e$ 为位置误差。

## 核心要点
1. 与阻抗控制（impedance control）对偶：阻抗控制以力为输出，导纳控制以位移为输出
2. 适合于力传感器精度较高的场景，对接触力进行低通滤波后再修正轨迹
3. 导纳参数（$M_d$, $B_d$, $K_d$）可由上层 VLA 模型动态调整，实现语义感知的柔顺行为
4. 在 [[PaCo-VLA]] 中，VLA 输出「导纳调度」作为任务级提案，高频无源性屏蔽层负责安全执行

## 代表工作
- [[PaCo-VLA]]：将 VLA 输出重定义为导纳调度，结合无源性屏蔽实现安全接触操作

## 相关概念
- [[MPC]]（另一种预测性控制策略）
- [[passivity theory]]（保证系统稳定的能量观点）
- [[Impedance Control]]（与导纳控制对偶）
