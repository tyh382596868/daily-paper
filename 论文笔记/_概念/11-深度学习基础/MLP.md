---
type: concept
aliases: [MLP, Multilayer Perceptron, 多层感知机]
---

# MLP

## 定义

多层感知机：由若干全连接层（Linear）与非线性激活（如 ReLU、GELU）堆叠而成的最基本前馈神经网络结构。在现代深度学习里通常作为**投影头 / projector**、[[Transformer]] 的 FFN 子层、或 reward / value head。

## 数学形式

$$
y = W_L\,\sigma(\dots\sigma(W_2\,\sigma(W_1 x + b_1) + b_2)\dots) + b_L
$$

其中 $\sigma$ 是激活函数（ReLU / GELU / SiLU 等）。

## 核心要点

1. **万能近似**: 给定足够宽度，可以近似任意连续函数（理论结果，实践中受优化限制）
2. **作为投影头**: 自监督方法中常用 1–3 层 MLP 把 backbone 表示投影到对比空间，[[LeWM]] 即用 1 层 MLP + [[Batch Normalization]] 作为 encoder 投影头
3. **Transformer FFN**: 标准 [[Transformer]] 中每个 attention 子层后接一个 2 层 MLP（hidden 通常 4× 模型维度）
4. **作为 probing**: 评估表示质量时用线性 / MLP probe 拟合下游属性（如 [[LeWM]] 的物理 probe）

## 相关概念

- [[Transformer]]
- [[Batch Normalization]]
- [[CLS Token]]
