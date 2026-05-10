---
type: concept
aliases: [Control Barrier Function, 控制障碍函数]
---

# CBF

## 定义
一种基于李雅普诺夫稳定性理论的安全约束方法，通过定义障碍函数来确保系统状态始终保持在安全集合内。

## 数学形式
$$\dot{h}(x) + \alpha(h(x)) \geq 0$$
其中 $h: \mathbb{R}^n \to \mathbb{R}$ 为障碍函数（$h(x) \geq 0$ 表示安全状态），$\alpha$ 为 class-K 函数。

## 核心要点
1. 将安全约束转化为 QP（二次规划）问题，可实时求解
2. 保证系统轨迹不逃出安全集合（前向不变性）
3. 在协作机器人、辅助遥操作（如 [[AssistDLO]]）中广泛使用

## 相关概念
- [[MPC]]
