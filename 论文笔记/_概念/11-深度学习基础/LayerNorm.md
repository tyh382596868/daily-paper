---
type: concept
aliases: [Layer Normalization, 层归一化, LayerNorm]
---

# LayerNorm

## 定义

对单样本沿**特征维度**做归一化的层，统计当前样本在 hidden 维度上的均值与方差，再做仿射变换。与 [[BatchNorm]] 沿 batch 维度统计不同，LayerNorm 不依赖 batch 大小，是 [[Transformer]] 系列模型的默认归一化方式。

## 数学形式

$$
\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2
$$

其中 $d$ 是 hidden 维度，$\gamma, \beta$ 是可学习的尺度和偏置。

## 核心要点

1. **样本内统计**: 与 batch 无关，适合 RNN / Transformer / 在线推理
2. **稳定训练**: 缓解梯度爆炸/消失，是大模型训练稳定的关键
3. **Pre-LN vs Post-LN**: Pre-LN（LN 在 attention/FFN 之前）训练更稳；Post-LN（LN 在残差之后）表达力略强
4. **与 [[RMSNorm]] 区别**: RMSNorm 不减均值，只用 RMS 缩放；更轻、效果接近
5. **与 [[AdaLN]] 区别**: AdaLN 把 $\gamma, \beta$ 改为条件依赖（如时间步、文本），是 DiT / [[MMDiT]] 的标配

## 代表工作

- 原始 [[Transformer]]: 在每个 sub-layer 后用 LN
- [[Geometry Adapter]] / [[Reasoning Adapter]]: 用 LN 缓冲 adapter 输出分布
- BERT、GPT-2 等绝大多数早期 LLM 默认使用

## 相关概念

- [[BatchNorm]]
- [[RMSNorm]]
- [[AdaLN]]
- [[Transformer]]
