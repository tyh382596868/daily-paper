---
type: concept
aliases: [BEAST policy, B-Spline Action Sequence Token]
---

# BEAST

## 定义

BEAST 是一种把机器人动作 chunk 用 **B-spline 参数化**的策略动作头：策略输出若干控制点（control points），通过固定基函数的 B-spline 重建完整动作序列。属于"参数化轨迹"路线的 [[VLA]] 动作头。

## 核心要点

1. **段数有限**：B-spline 控制点数决定表达能力，高频细节需更多控制点；
2. **天然平滑**：B-spline 是 $C^{k}$ 连续（取决于阶数），对 [[Jerk]] 友好；
3. **离散仍然存在**：控制点是离散的，本质仍是"分段表示"；
4. **是 [[NIAF]] 的主要对比对象**：NIAF 用连续函数 [[SIREN]] 完全替代分段 B-spline。

## 代表工作

- **BEAST**：在 [[CALVIN]] / [[LIBERO]] 上取得强基线
- [[NIAF]]：在同样设置下用连续函数表示超过 BEAST

## 相关概念

- [[Action Chunking]]
- [[Neural Implicit Action Field]]
- [[FLOWER]]
- [[FAST]]
