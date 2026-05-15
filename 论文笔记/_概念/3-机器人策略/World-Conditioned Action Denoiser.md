---
type: concept
aliases: [World-Conditioned Action Denoiser, 世界条件动作去噪器]
---

# World-Conditioned Action Denoiser

## 定义

[[DAWN]] 中负责**根据当前 latent 与预测未来 latent 生成动作序列**的子模块。结构上是一个 [[DiT]] 风格的扩散去噪器，承担两种角色：proposal 生成（首轮，无未来条件）与 refinement 修正（之后每轮，看上一步动作 + 该动作触发的未来）。

## 数学形式

Proposal:
$$
a_{1:H}^{(0)} = G_\phi(q_{prop}, c, z)
$$

Refinement:
$$
a_{1:H}^{(r+1)} = G_\phi(q_{ref}^{(r)}, c, z_{future}^{(r)}, a_{1:H}^{(r)})
$$

## 核心要点

1. **双角色设计**: 用 role-specific query embedding $q_{prop}$ / $q_{ref}^{(r)}$ 区分两种调用
2. **以预测未来为条件**: 这是 [[WAIM]] 中 "world → action" 方向的核心载体
3. **保留上一轮动作**: 类似 self-conditioning，使迭代具有连续性
4. **扩散范式**: [[DiT]] 结构便于条件注入，且支持 trajectory interactive refinement（除从 scratch 生成外还可改写已有轨迹）

## 代表工作

- [[DAWN]]: 提出此组件，与 [[World Predictor]] 递归互修

## 相关概念

- [[WAIM]]
- [[World Predictor]]
- [[DiT]]
- [[Diffusion Policy]]
- [[递归交互推理]]
