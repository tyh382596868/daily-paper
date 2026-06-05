---
type: concept
aliases: [可供性退火]
---

# Affordance Annealing（可供性权重退火）

## 定义

AffordanceVLA 的下游微调技巧：Stage III 时把 affordance loss 权重 $\lambda_{afd}$ 从 0.5 降到 0.15，让动作损失主导，从而更好适配特定环境的精细控制需求。

## 核心要点

1. **课程式退火**: Stage I 单独训 affordance → Stage II $\lambda_{afd}=0.5$ → Stage III $\lambda_{afd}=0.15$。
2. **避免可供性偏置**: 下游环境可能与预训练 affordance 标注分布不同，权重退火防止其干扰控制学习。
3. **动作主导**: Stage III 中 $\lambda_{act}=1.0$，$\lambda_{afd}=0.15$，比例约 6.7:1。
4. **兼顾迁移与精度**: 既保留可供性的 OOD 泛化收益，又保证控制精度。

## 代表工作

- [[AffordanceVLA]]

## 相关概念

- [[Affordance Forecasting]]
- [[Curriculum Learning]]
- [[Sample Efficiency]]
