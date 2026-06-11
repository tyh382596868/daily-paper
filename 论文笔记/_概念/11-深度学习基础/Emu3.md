---
type: concept
aliases: [Emu3, Emu3-Base, Emu3-Gen, BAAI Emu3]
---

# Emu3

## 定义

BAAI 提出的统一多模态基础模型，通过将图像、文本统一为离散 token 序列进行自回归建模，实现图像生成、视觉理解和文本生成的一体化，是 UniVLA 的骨干架构基础。

## 核心要点

1. **统一 token 空间**: 图像用 VQ-VAE（codebook 32,768）离散化，与文本 token 共享词表（vocab 184,622）
2. **Encoder-Free**: 无需独立图像编码器，直接以 visual token 作为输入
3. **MoE 扩展（Emu3-MoE）**: 混合专家架构，提升参数效率
4. **模型规格**: 32 层 Transformer，hidden size 4096，32 heads，GQA（8 KV heads），~8B 参数

## 模型配置

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 184622
}
```

## 代表工作

- Emu3: Next-Token Prediction is All You Need（BAAI, 2024）
- [[UniVLA]]: 基于 Emu3-MoE 构建的统一 VLA 模型

## 相关概念

- [[VQ Tokenizer]]: Emu3 视觉分词组件（Emu3-VisionTokenizer）
- [[Emu3-MoE]]: Emu3 的 MoE 扩展版本
- [[Autoregressive Transformer]]: Emu3 的架构类型
