---
type: concept
aliases: [Mixture of Experts, 混合专家模型]
---

# MoE

## 定义
一种条件计算架构，将模型参数分组为多个"专家"，通过门控网络动态路由不同输入到不同专家，在增加参数量的同时控制计算成本。

## 数学形式
$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$
其中 $G(x)$ 为门控函数（通常选 Top-K），$E_i$ 为第 $i$ 个专家网络。

## 核心要点
1. Sparse MoE：每个 token 只激活 Top-K 个专家（通常 K=1 或 2）
2. 在 LLM（Mixtral、DeepSeek-MoE）和 VLA（VLA-GSE）中广泛使用
3. 负载均衡（load balancing loss）是训练关键

## 代表工作
- [[VLA-GSE]]: 将 MoE 用于 VLA 的 PEFT
- [[HANDOFF]]: MoE 学生策略，三路 Expert 分别由 WBC/Locomotion/跌倒恢复 Teacher 通过 context 条件化 KL 蒸馏训练

## 相关概念
- [[LoRA]]
- [[SVD]]
