---
type: concept
aliases: [推理 Adapter, 推理适配器]
---

# Reasoning Adapter

## 定义
一个**轻量 MLP 投影器**，把 student 路径（带 action prompt）下的 [[Reasoning Anchor Token|reasoning anchor]] 隐藏态映射到 teacher 路径（带 explicit reasoning prompt）的潜空间，是 [[Latent Distillation]] 中不可缺少的"分布桥梁"。

## 数学形式

$$
\mathcal{L}_{reasoning} = 1 - \mathcal{S}\bigl(H^R_{teacher},\; \mathcal{R}(\hat{H}^R_{student})\bigr)
$$

其中 $\mathcal{R}$ 即 reasoning adapter，结构通常为 2-3 层 MLP。

## 核心要点
1. **缓解 prompt 分布偏移**: teacher 与 student 的输入 prompt 不同，hidden state 分布不同，直接做 cosine 距离会震荡
2. **可学习**: 与 anchor token 一起训练
3. **推理时**: 仅用于 action 路径的投影到几何/动作空间，teacher 分支整体丢弃
4. **与 Geometry Adapter 配对**: 一个对应几何路径，一个对应推理路径，二者输出在动作端相加融合

## 代表工作
- [[3DThinkVLA]]: 首次提出 reasoning adapter 与 reasoning anchor 配对设计

## 相关概念
- [[Latent Distillation]]
- [[Reasoning Anchor Token]]
- [[Geometry Adapter]]
- [[LoRA]]
