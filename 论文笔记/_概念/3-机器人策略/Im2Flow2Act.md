---
type: concept
aliases: []
---

# Im2Flow2Act

## 定义

Cascaded WAM 的几何抽取（Geometric Extraction）代表：从生成的视频中**计算光流**，再用闭式几何把光流转成末端执行器轨迹，无需训练 action head。

## 核心要点

1. **闭式几何动作抽取**: 替代学习式 [[逆动力学]]。
2. **零额外动作监督**: 光流→姿态变换是几何运算，不需要 action label。
3. 属于 [[Cascaded WAM|Cascaded WAM]] 的 Explicit / Geometric 子类。

## 代表工作

- [[WAM-Survey]] 中作为几何抽取路线的开端。

## 相关概念

- [[3DFlowAction]]
- [[World Action Model]]
