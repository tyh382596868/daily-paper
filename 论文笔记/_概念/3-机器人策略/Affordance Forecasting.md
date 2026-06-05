---
type: concept
aliases: [结构化可供性预测, Structured Affordance Forecasting]
---

# Affordance Forecasting（结构化可供性预测）

## 定义

把"可供性"显式拆为多个结构化预测任务（如对象级 grounding、2D 区域、3D 形状、位姿等），作为 VLA 模型的可学习中间表征，桥接感知和动作生成。

## 核心要点

1. **结构化拆解**: 不再用单一 affordance map，而是分多个互补维度（Which/Where/How）。
2. **可学习中间表征**: 区别于把可供性当外部输入，可供性预测本身参与训练，与下游动作 co-train。
3. **优势**: 显式可解释 + 提升数据效率 + 改善 OOD 泛化。
4. **典型范式**: Stage I 单独预训练可供性专家，Stage II 与动作专家联合微调。

## 代表工作

- [[AffordanceVLA]]: 系统化提出 Which/Where/How 三层范式

## 相关概念

- [[Affordance]]
- [[Affordance Generation Expert]]
- [[VLA]]
- [[MoT]]
