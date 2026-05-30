---
type: concept
aliases: [VoE, 期望违反范式, Violation of Expectation]
---

# Violation of Expectation (VoE)

## 定义

认知科学经典范式：通过让被试观察"违反物理/因果直觉"的事件，观察其反应（如注视时间增加）来判断其是否理解相应的世界知识。最早用于婴儿认知研究。

## 核心要点

1. 提供了一种**无需主动回答**的探针方法
2. 在 DL 中被复用：通过模型对"违反预期"输入的反应（如更高 loss）来探测其世界知识
3. [[YoCausal]] 把时间反转视为天然的 VoE 样本

## 代表工作

- [[YoCausal]]: 用 VoE 范式构造 [[Counterfactual|反事实]] 评测
- Riochet et al. "IntPhys" (TPAMI 2020): 用 VoE 测物理理解
- Piloto et al. "Probing Physics Knowledge" (Nat Hum Behav 2022)

## 相关概念

- [[Counterfactual]]
- [[World Model]]
- [[Arrow of Time]]
