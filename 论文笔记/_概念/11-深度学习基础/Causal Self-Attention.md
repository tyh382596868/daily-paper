---
type: concept
aliases: [因果自注意力, Causal Mask, Causal Attention]
---

# Causal Self-Attention

## 定义

**Causal Self-Attention** 是在自回归生成中使用的注意力变体：通过下三角 mask 强制每个 token 只能 attend 到自己及其之前的 token，保证训练阶段并行计算与推理阶段因果生成的一致性。

## 核心要点

1. **下三角 mask**: $M_{ij} = -\infty$ if $j > i$
2. **训练 / 推理一致**: 训练时并行计算所有位置，推理时逐 token 生成
3. **应用**: GPT 系、[[VLM]]、[[VLA]]、[[Cosmos3]] 的 AR 塔
4. **vs Bidirectional**: 编码任务（如 BERT、扩散去噪）使用双向 attention 可看全局

## 代表工作

- [[Cosmos3]]: AR 推理塔内部使用 causal mask
- 所有 GPT / Qwen / Llama 系列

## 关联

- [[Next-Token Prediction]]: causal attention 的训练目标
- [[Cross-Attention]]: 对比
- [[Transformer]]: 基础架构
