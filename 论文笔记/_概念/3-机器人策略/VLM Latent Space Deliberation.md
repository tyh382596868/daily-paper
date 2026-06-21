---
type: concept
aliases: [VLM 潜空间审议, 潜空间推理]
---

# VLM Latent Space Deliberation

## 定义
在视觉-语言模型的潜在特征空间内进行多轮 deliberation（审议），无需将推理过程解码为文字 token。

## 数学形式
$$z^{(k+1)} = f_{deliberate}(z^{(k)}, o, l)$$
其中 $z^{(k)}$ 为第 $k$ 轮潜在状态，多轮迭代收敛到最优规划。

## 核心要点
1. 避免 CoT 解码 token 带来的延迟开销
2. 潜空间操作保留了 VLM 的隐式知识
3. 支持在低延迟约束下进行隐式多步规划

## 代表工作
- [[PearlVLA]]: 在 VLM latent space 做 progressive deliberation

## 相关概念
