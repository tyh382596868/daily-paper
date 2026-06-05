---
type: concept
aliases: [InternData-A1]
---

# InternData-A1

## 定义

一个大规模合成机器人交互数据集，覆盖多种操作任务，用于 VLA 模型的大规模联合预训练；AffordanceVLA 在其上做 Stage II 联合训练并自动生成超过 100K 个 affordance 标注。

## 核心要点

1. **合成数据**: 利用仿真生成多样化机器人操作轨迹。
2. **自动可供性标注**: AffordanceVLA 在其上构建自动 affordance 标注 pipeline，>100K 标签。
3. **Stage II 主数据**: 用于联合动作和可供性预训练。
4. **大规模联合预训练**: 数据规模显著大于 Stage I 的 grounding 数据，是性能关键。

## 代表工作

- [[AffordanceVLA]]: Stage II 主数据

## 相关概念

- [[DROID]]
- [[LIBERO]]
- [[CALVIN]]
- [[Affordance Forecasting]]
