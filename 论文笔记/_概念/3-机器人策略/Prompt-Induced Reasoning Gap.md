---
type: concept
aliases: [Prompt 引起的推理空缺, 提示路径推理鸿沟]
---

# Prompt-Induced Reasoning Gap

## 定义
[[3DThinkVLA]] 提出并诊断的现象：在 VLA + VLM 协同训练过的模型上，当 prompt 从"explicit reasoning prompt"切换到"action prompt"时，模型**关闭/绕开**刚学到的空间推理能力，转而走 [[Action Shortcut|动作捷径]]，导致 3D 协同训练的收益无法迁移到真实动作预测。

## 数学形式

定义 anchor token 在两条路径上的潜空间相似度：
$$
\mathrm{sim}(t) = \mathcal{S}\bigl(H^R_{teacher}(t),\; H^R_{student}(t)\bigr)
$$

vanilla co-training 下 $\mathrm{sim}(t)$ 训练中后期下降到 < 0.5（[[3DThinkVLA]] Figure 5b）；本文方法通过 [[Latent Distillation|隐式蒸馏]]使 $\mathrm{sim}(t) > 0.8$。

## 核心要点
1. **现象**: 协同训练后 VLM 在"3D QA prompt"下空间推理正确，但同一模型在"action prompt"下退化为忽略空间细节
2. **机理**: 自回归模型对 prompt 模板高度敏感，action prompt 触发"直接出动作"的捷径分布
3. **可量化**: 用 anchor hidden state 的相似度曲线诊断
4. **解决方案**: 把 explicit reasoning prompt 的 anchor hidden state 蒸馏到 action prompt 的 anchor hidden state（[[Reasoning Anchor Token]] + [[Reasoning Adapter]]）

## 代表工作
- [[3DThinkVLA]]: 首次明确诊断并提出解决方案

## 相关概念
- [[Action Shortcut]]
- [[Reasoning Anchor Token]]
- [[Latent Distillation]]
- [[Causal Confusion Paradox]]
