---
type: concept
aliases: [Wrist-View Surfel Memory, 腕部视角 Surfel 记忆]
---

# W-VMem

## 定义

W-VMem（Wrist-View-centered surfel-indexed Memory）是 [[Mem-World]] 提出的 4D 几何记忆模块，将历史观测锚定到时间演化的表面元素（[[Surfel]]）上，通过综合几何可见性、任务相关性和时间衰减三项指标检索与当前动作最相关的历史帧。

## 数学形式

评分函数：

$$
\mathrm{score}(s, t) = \frac{\langle \mathbf{n}_s,\, \bar{\mathbf{v}}_w \rangle}{1 + d_s} \cdot \ln(e + m_s) \cdot \left[\lambda_{\min} + (1 - \lambda_{\min})\, 2^{-\frac{T - t}{H}}\right]
$$

Surfel 定义：$s_k = (p_k, n_k, r_k, t_k, m_k)$，相比 [[VMem]] 额外增加时间戳 $t_k$ 和操作物标志 $m_k$。

## 核心要点

1. **腕部视角中心**: 仅用腕部视角帧更新 surfel，防止第三视角反复刷新时间戳，保留时间关联
2. **前向运动学推导位姿**: 由策略输出的关节动作 + 固定几何变换推导未来腕部位姿，无需独立位姿估计
3. **三因子评分**: 几何可见性 × 任务相关性 × 时间衰减，各因子独立可解释
4. **NMS 去冗余**: 检索后施加 [[Non-Maximum Suppression]] 避免重复访问区域提供冗余上下文

## 代表工作

- [[MemWorld]]: 提出 W-VMem 并验证其在机器人操作世界模型中的效果（策略评估相关性 +14.5%）

## 相关概念

- [[Surfel]]
- [[VMem]]
- [[Action-Conditioned World Model]]
- [[Forward Kinematics]]
