---
type: concept
aliases: [AdaLN, Adaptive Layer Normalization, 自适应层归一化]
---

# AdaLN

## 定义

Adaptive Layer Normalization 是把外部条件信号（如时间步、动作、文本）注入 [[Transformer]] 的一种轻量机制：用一个小 MLP 把条件映射为缩放与平移参数 $\gamma,\beta$，对 [[Transformer]] 每层归一化后的隐藏状态做仿射调制。被 [[DiT]] 推广为 DiT 中的核心条件化方式。

## 数学形式

$$
\text{AdaLN}(h, c) = \gamma(c)\odot \text{LN}(h) + \beta(c)
$$

其中 $h$ 是隐藏状态、$c$ 是条件向量、$\gamma(c),\beta(c)$ 由 MLP 输出。

## 核心要点

1. **比拼接更优雅**: 不增加 token 长度，对每层都做条件化
2. **零初始化**: $\gamma,\beta$ 通常零初始化（"AdaLN-Zero"），训练初期层近似 identity，避免条件信号"带歪"训练
3. **DiT / SiT 标配**: 把扩散时间步与类别条件统一塞进 AdaLN
4. **在世界模型中**: [[LeWM]] 把动作 $a_t$ 通过 AdaLN 注入 predictor 每一层

## 代表工作

- [[DiT]]: Diffusion Transformer，AdaLN-Zero 的代表
- [[LeWM]]: 用 AdaLN 做 predictor 的动作条件化

## 相关概念

- [[Transformer]]
- [[DiT]]
- LayerNorm
