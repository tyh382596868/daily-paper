---
type: concept
aliases: [数据效率, Data Efficiency]
---

# Sample Efficiency（样本/数据效率）

## 定义

模型用少量数据即达到给定性能的能力。在机器人/VLA 领域，由于真实数据标注成本高，数据效率是评判方法实用性的关键指标。

## 核心要点

1. **衡量方式**: 给定相同算法/对比基线，达到目标成功率所需的 training data 量。
2. **提升手段**:
   - 引入结构化中间表征（如 [[Affordance Forecasting]]）
   - 大规模预训练
   - 数据增广
   - 模仿学习 + 强化学习混合
3. **AffordanceVLA 案例**: 仅 40% 训练数据即超过 vanilla [[Pi05|π₀]] 满数据性能（LIBERO ~92%）。
4. **OOD 泛化关联**: 高样本效率通常伴随强 OOD 泛化能力。

## 代表工作

- [[AffordanceVLA]]: 通过可供性中间表征显著提升数据效率

## 相关概念

- [[Affordance Forecasting]]
- [[Curriculum Learning]]
- [[Pre-training]]
