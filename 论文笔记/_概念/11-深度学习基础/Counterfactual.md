---
type: concept
aliases: [反事实, 反事实样本, Counterfactual]
---

# Counterfactual

## 定义

"如果当时不是 A 而是 B 会怎样？"——通过构造一个与真实世界仅在关键变量上不同的样本，用于探测模型的因果推理能力。

## 核心要点

1. **强反事实**：在因果链关键节点替换原因，观察结果是否变化
2. **弱反事实**：替换无关变量，结果应不变（用于鲁棒性测试）
3. [[YoCausal]] 利用**时间反转**作为零成本反事实：保留所有视觉内容，仅破坏时序因果
4. 反事实样本通常需要保持分布内（in-distribution），否则会与 OOD 检测混淆

## 代表工作

- [[YoCausal]]: 时间反转作为反事实
- Pearl 因果阶梯（关联 → 干预 → 反事实）

## 相关概念

- [[Causality Cognition Index]]
- [[Violation of Expectation]]
- [[Arrow of Time]]
