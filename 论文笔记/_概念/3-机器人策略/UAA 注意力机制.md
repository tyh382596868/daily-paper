---
type: concept
aliases: [UAA Attention, Understanding-Affordance-Action Attention, UAA 渐进注意力]
---

# UAA 注意力机制（Understanding–Affordance–Action Progressive Attention）

## 定义

AffordanceVLA 在 MoT 三专家间引入的渐进式因果注意力机制：每个专家内部用双向注意力，跨专家用因果掩码，确保信息严格按 Understanding → Affordance → Action 单向流动。

## 数学形式

$$
\text{Attn}_{\text{afd}} = \text{Attention}(Q_{gen}, K_{und}, V_{und})
$$

$$
\text{Attn}_{\text{act}} = \text{Attention}(Q_{act}, [K_{und}; K_{gen}], [V_{und}; V_{gen}])
$$

## 核心要点

1. **专家内双向**: 同一专家的 token 之间无掩码，最大化上下文融合。
2. **跨专家因果**: Affordance 只能 attend Understanding；Action 可 attend 两者；上游看不到下游。
3. **防止动作泄漏**: 避免"未来动作"反向污染感知/规划阶段的预测。
4. **必要性验证**: AffordanceVLA 消融中改成 block-wise（无跨专家）→ LIBERO 95.8 → 90.3。

## 代表工作

- [[AffordanceVLA]]: 首次提出该机制

## 相关概念

- [[MoT]]
- [[Understanding Expert]]
- [[Affordance Generation Expert]]
- [[Action Expert]]
- [[Causal Attention]]
