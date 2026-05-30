---
type: concept
aliases: [Causal Forcing Plus Plus, CF++]
---

# Causal Forcing++

## 定义

[[Causal Forcing]] 的**在线变体**：用 [[Causal Consistency Distillation]] 替换原本需要离线 ODE 数据的 Stage 2，整套 pipeline 不再依赖大规模离线监督，存储和时间成本显著降低。

## 数学形式

整体阶段流程与 [[Causal Forcing]] 相同，Stage 2 损失见 [[Causal Consistency Distillation]]，Stage 3 仍是 [[Asymmetric DMD]]。

## 核心要点

1. **省存储**：不需要预生成数 TB 级的 ODE 监督轨迹。
2. **训练即蒸馏**：在线一致性约束让学生在训练过程中自我压缩。
3. 早期需要 warm-up（或 ODE 短暂初始化）以避免坍塌。

## 代表工作

- [[minWM]]: Stage 2B 选择的路径，与 [[Causal Forcing]] 并列作为 pipeline 的两条可选支线。

## 相关概念

- [[Causal Forcing]]
- [[Causal Consistency Distillation]]
- [[Asymmetric DMD]]
