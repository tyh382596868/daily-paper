---
type: concept
aliases: [Per-Layer KV Connection, per-layer KV, 逐层KV连接]
---

# 逐层 KV 连接（Per-Layer KV Connection）

## 定义

逐层 KV 连接是一种将大型视觉-语言模型（VLM）的层级特征注入下游专家模型的架构设计：动作专家的第 $\ell$ 层[[交叉注意力]]直接访问 VLM 第 $\ell$ 层的键值（KV）缓存，而非共享单一全局上下文。

## 数学形式

$$
\tilde{K}_\ell = \text{reshape}(P_K K^\text{vlm}_\ell), \quad \tilde{V}_\ell = \text{reshape}(P_V V^\text{vlm}_\ell)
$$

$$
\text{CA}(Q_\ell, \tilde{K}_\ell, \tilde{V}_\ell) = \text{softmax}\!\left(\frac{Q_\ell \tilde{K}_\ell}{\sqrt{d_h}}\right) \tilde{V}_\ell
$$

## 核心要点

1. **层级对齐**：VLM 第 $\ell$ 层的特征对应动作专家第 $\ell$ 层，在等深度进行特征交互
2. **维度适配**：通过可学习线性投影 $P_K, P_V$ 对齐 VLM 与专家的注意力维度
3. **对比传统方法**：传统单一[[交叉注意力]]只从 VLM 最后一层提取全局上下文，逐层连接提供更丰富的层级信息
4. **推理优化**：VLM KV 缓存可跨时间步复用，降低重复计算

## 代表工作

- [[MolmoAct2]]：首次在 VLA 中引入逐层 KV 连接，用于连接 Molmo2-ER 骨干与流匹配动作专家

## 相关概念

- [[交叉注意力]]
- [[Flow Matching]]
- [[VLA]]
- [[自注意力]]
