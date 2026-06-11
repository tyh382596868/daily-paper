---
type: concept
aliases: [Visual Token, 视觉词元, Vision Token, 图像Token]
---

# 视觉 Token

## 定义

将图像或视频帧通过 VQ-VAE 等向量量化编码器转换为离散符号（Token）的表示，使视觉内容可在自回归语言模型框架中与文本 Token 统一建模。

## 核心要点

1. **空间压缩**: 典型配置将图像压缩 8× 或 16×，每帧生成数百到数千个视觉 Token（Emu3 为 4096/帧）。
2. **离散化**: 通过 VQ-VAE 码本将连续视觉特征映射为整数索引，纳入文本词表扩展或独立词表。
3. **Next-Token Prediction 监督**: 视觉 Token 可作为自回归预测目标（如世界模型后训练），也可作为条件输入。
4. **信息保留 vs. 序列长度**: 更多 Token 保留更多细节，但序列变长推理代价显著增加（UniVLA 4096 Token/帧面临的主要挑战）。

## 代表工作

- [[Emu3]]（BAAI, 2024）: 每帧 4096 视觉 Token，使用 SBER-MoVQGAN
- [[UniVLA-ICLR2026]]（BAAI, 2025）: 继承 Emu3 视觉 Tokenizer，扩展到机器人策略
- Chameleon（Meta）: 图文统一 Token 自回归生成

## 相关概念

- [[VQ-VAE]]
- [[动作 Token]]
- [[自回归 Transformer]]
- [[Emu3]]
- [[Next-Token Prediction]]
