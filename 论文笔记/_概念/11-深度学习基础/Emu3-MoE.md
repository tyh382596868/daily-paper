---
type: concept
aliases: [Emu3 MoE, Emu3MoE, BAAI Emu3 MoE]
---

# Emu3-MoE

## 定义

Emu3 的混合专家（Mixture of Experts）扩展版本，在 Emu3 基础架构上引入 MoE 层以提升参数效率，是 UniVLA 使用的核心骨干模型。

## 核心要点

1. **MoE 架构**: 用多个专家 FFN 替代标准 FFN，每次推理只激活部分专家（稀疏激活）
2. **动作专家扩展**: `action_config` 中额外定义了 2 层小型 Transformer 作为动作解码头（hidden 4096，FFN 2048）
3. **双损失权重**: 视觉生成损失权重可配置（`vision_loss_weight`），动作损失权重独立设置（5.0）
4. **Flash Attention 2**: 支持 FA2 加速，训练使用 BF16

## 配置参数（policy 阶段）

| 参数 | 值 |
|------|-----|
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA) |
| intermediate_size | 14336 |
| vocab_size | 184,622 |
| max_position_embeddings | 1700（策略）/ 6400（预训练）|

## 代表工作

- [[UniVLA]]: 以 Emu3-MoE 为骨干的统一 VLA 模型

## 相关概念

- [[Emu3]]: 基础版本
- [[VQ Tokenizer]]: 配套视觉分词器
- [[FAST Action Tokenizer]]: 配套动作分词器
- [[Autoregressive Transformer]]: 架构类型
