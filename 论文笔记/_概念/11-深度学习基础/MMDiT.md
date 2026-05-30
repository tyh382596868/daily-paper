---
type: concept
aliases: [Multi-Modal Diffusion Transformer]
---

# MMDiT

## 定义
**Multi-Modal [[扩散变换器|Diffusion Transformer]]** —— 把文本与视觉 token 拼接进同一注意力流的[[扩散变换器]]变体，最早在 Stable Diffusion 3 提出，并广泛被 HY-Video、HY1.5-TI2V 等沿用。每条模态有独立的 Q/K/V projection，但在 [[自注意力]] 中合并计算。

## 数学形式

$$
\text{Attn}([Q_{\text{text}}; Q_{\text{img}}],\ [K_{\text{text}}; K_{\text{img}}],\ [V_{\text{text}}; V_{\text{img}}])
$$

## 核心要点

1. **联合自注意力**: 文本与视觉双向交互，比 cross-attention 范式更对称。
2. **模态独立投影**: 各模态保留独立 Q/K/V/AdaLN，但共享主干注意力。
3. **下游 DiT 主流**: HY1.5-TI2V、SD3、FLUX 均属 MMDiT 系。

## 代表工作

- [[minWM]]: HY1.5-TI2V-8B 是 MMDiT 架构，与 Cross-Attention 条件的 [[Wan2.1]] 形成对照。
- [[Wan2.2]] / [[FLUX]] 等

## 相关概念

- [[扩散变换器|DiT]]
- [[自注意力]]
- [[Cross-Attention]]
