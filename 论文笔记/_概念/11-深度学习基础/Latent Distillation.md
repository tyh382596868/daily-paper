---
type: concept
aliases: [潜空间蒸馏, 隐式蒸馏, Hidden-State Distillation]
---

# Latent Distillation

## 定义
[[Knowledge Distillation|知识蒸馏]] 的一种特殊形式：不对最终 logits 也不对显式输出 token 做匹配，而是直接在某个**潜在表征**（hidden state、CLS token、anchor token）上让 student 逼近 teacher。

## 数学形式

$$
\mathcal{L}_{latent} = 1 - \mathcal{S}\bigl(h_T,\; \mathcal{R}(h_S)\bigr)
$$
其中：
- $h_T = sg(f_\theta(\cdot; \text{teacher prompt}))$ 是 teacher hidden state（stop-gradient）
- $h_S = f_\theta(\cdot; \text{student prompt})$ 是 student hidden state
- $\mathcal{R}$ 是把 student 投到 teacher 潜空间的可学习 projector（避免 prompt 路径差异带来的分布偏移）
- $\mathcal{S}$ 通常为 [[Cosine Similarity|余弦相似度]]

## 核心要点
1. **避免 logits 蒸馏的瓶颈**: 输出空间维度过低时 logits 监督信号弱
2. **支持非生成式任务迁移**: 例如把 reasoning 能力蒸馏到 action 预测路径
3. **Online 共享 backbone**: teacher 与 student 共用一组参数，仅 prompt / token 不同，省显存
4. **必备 projector**: student 与 teacher 的输入分布不同，需要 adapter 缓冲，否则直接匹配会震荡

## 代表工作
- [[3DThinkVLA]]: 用 reasoning anchor token 做 online latent distillation
- DistilBERT: 用 hidden state MSE 蒸馏

## 相关概念
- [[Knowledge Distillation]]
- [[Reasoning Anchor Token]]
- [[Cosine Similarity]]
