---
type: concept
aliases: [Where2Act, 2D 可供性损失]
---

# Where2Act Loss

## 定义

AffordanceVLA 中用于学习"在物体的哪个区域可交互"的损失：用 Transformer decoder 预测 2D affordance map，用 pixel-wise [[Binary Cross-Entropy]] 监督。

## 数学形式

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1 - M_i) \log(1 - \sigma(\hat{y}_i)) \right]
$$

## 核心要点

1. **像素级监督**: $M_i \in \{0, 1\}$ 是可交互区域 ground truth。
2. **指令感知**: 同一物体在不同语言指令下产生不同 affordance map。
3. **轻量 decoder**: Transformer decoder + spatial PE，参数量低。
4. **消融影响最大**: 去掉 Where2Act → LIBERO 95.8 → 93.2（−2.6），三个 affordance 模块中最重要。

## 代表工作

- [[AffordanceVLA]]
- [[Where2Act 系列]]: 早期 2D affordance map 路线

## 相关概念

- [[Affordance]]
- [[Affordance Forecasting]]
- [[Binary Cross-Entropy]]
- [[Affordance Generation Expert]]
- [[Which2Act Loss]]
