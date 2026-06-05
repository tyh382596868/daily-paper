---
type: concept
aliases: [可供性生成专家, Affordance Expert]
---

# Affordance Generation Expert

## 定义

AffordanceVLA 中专门负责结构化可供性预测的 MoT 专家，将语义对齐表征转换为可指导动作的几何/语义线索（Which/Where/How 三层 token）。

## 核心要点

1. **三类 token**: 输出 which / where / shape / layout 四组 token，对应对象 grounding、2D 区域、3D 形状、10-DoF 位姿。
2. **MoT 专家**: 与 [[Understanding Expert]] 和 [[Action Expert]] 并列，独立 FFN，共享 attention KV（受 UAA 渐进掩码约束）。
3. **多任务监督**: 由 4 个损失（MSE + BCE + 扩散 + Smooth-L1）联合监督。
4. **Stage I 单独训**, Stage II 与动作 co-train，Stage III 退火权重让控制主导。

## 代表工作

- [[AffordanceVLA]]: 提出该专家设计

## 相关概念

- [[Understanding Expert]]
- [[Action Expert]]
- [[MoT]]
- [[UAA 注意力机制]]
- [[Which2Act Loss]]
- [[Where2Act Loss]]
- [[How2Act Shape Loss]]
- [[How2Act Layout Loss]]
