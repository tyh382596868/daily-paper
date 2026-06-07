---
type: concept
aliases: [推理锚点 token, Reasoning Anchor, Spatial Anchor Token]
---

# Reasoning Anchor Token

## 定义
插入在 task instruction 之后、action 指令之前的**可学习 token** $\tau_R$，作用是承载"空间/语义推理"的潜在表征，让自回归 VLM 在生成动作前必先"经过"一个推理状态。可视为 [[CoT|思维链]]的隐式压缩版本——把整段推理压缩到一个 token 上。

## 数学形式

带 anchor 的动作预测：
$$
\hat{A}_t = \pi_\Theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)
$$

Teacher / Student 双路径下的 anchor 隐藏态：
$$
H^R_{teacher} = sg(f_\theta(I_t, L_{task}, L^{teacher}, \tau_R))
$$
$$
\hat{H}^R_{student} = f_\theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)
$$

## 核心要点
1. **位置固定**: $L_{task}$ 之后、$L_{action}$ 之前——这是关键，让 anchor 处于"已知任务、未提动作"的状态
2. **协同训练时作为首输出 token**: 在 VLM 训练步上要求 $\tau_R$ 是第一个生成 token，强制自回归条件依赖空间先验
3. **可被蒸馏**: 用 cosine 距离把 student 的 anchor hidden state 拉向 teacher（带 explicit reasoning prompt）
4. **推理时零额外开销**: 仅是一个 token 的 forward
5. **与 CLS Token 类比**: 都是聚合表征的"占位 token"，但 anchor 同时承担蒸馏目标的角色

## 代表工作
- [[3DThinkVLA]]: 首次将 reasoning anchor 引入 VLA latent distillation
- [[COCONUT]]: 用连续 thought tokens 取代显式 CoT 文本

## 相关概念
- [[Latent Distillation]]
- [[Prompt-Induced Reasoning Gap]]
- [[CoT]]
- [[CLS Token]]
