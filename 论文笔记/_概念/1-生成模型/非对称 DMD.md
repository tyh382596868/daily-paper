---
type: concept
aliases: [Asymmetric DMD, asymmetric Distribution Matching Distillation]
---

# 非对称 DMD

## 定义
[[DMD|Distribution Matching Distillation]] 的变体：教师与学生在**架构/条件结构上不对称**——典型用法是 frozen 双向扩散教师 + 因果 [[自回归]] 学生，通过 $s_{\text{real}}$ 与 $s_{\text{fake}}$ 的差异做 score divergence 蒸馏，把 few-step AR 学生的隐分布拉向多步双向教师分布。

## 数学形式

$$
\nabla_\theta\ \mathbb{E}_t\bigl[D_{KL}\bigl(p_{\theta,t}(\tilde x_t)\ \|\ p_{\text{data},t}(\tilde x_t)\bigr)\bigr]
= -\,\mathbb{E}_{\tilde x,t,\tilde x_t}\Bigl[\bigl(s_{\text{real}}(\tilde x_t,t) - s_{\text{fake}}(\tilde x_t,t)\bigr)\,\frac{\partial \tilde x}{\partial \theta}\Bigr]
$$

## 核心要点

1. **架构非对称**: 教师双向、学生自回归，结构不同但仍可用 score divergence 对齐。
2. **条件全程注入**: 所有 score model 都接收相机/动作条件，避免蒸馏中丢失控制能力。
3. **few-step 友好**: 与 [[Consistency Distillation]] / [[Causal ODE 初始化]] 串联，可压到 4 步内。

## 代表工作

- [[minWM]]: 在 Stage 3 用非对称 DMD 把 chunk-wise AR 学生压到 4 step，HY1.5 首帧延迟 1.041s → 3.46ms。

## 相关概念

- [[DMD]]
- [[Score Function]]
- [[Consistency Distillation]]
- [[Causal ODE 初始化]]
