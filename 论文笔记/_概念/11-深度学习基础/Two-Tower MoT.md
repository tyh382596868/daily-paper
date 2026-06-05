---
type: concept
aliases: [Two-Tower Mixture-of-Transformers, 双塔 MoT, Two-Tower Architecture]
---

# Two-Tower MoT

## 定义

**Two-Tower Mixture-of-Transformers** 是一种在同一 Transformer 栈内部为不同模态/任务保留两套独立专家参数的架构。两个"塔"分别处理离散自回归 token（AR 塔）和连续扩散 latent（DM 塔），但共享 attention 投影空间，从而在一次前向中完成"理解 → 生成"。

## 核心要点

1. **参数解耦 + Attention 共享**: 每层的 FFN（甚至 attention 的 QKV 投影）按模态路由到独立 expert，但 softmax attention 在两塔间联合计算
2. **单向条件流**: AR 塔可独立运行（VLM 模式），DM 塔必须以 AR 输出为条件（生成模式），通过 attention mask 实现
3. **比单塔多模态保真度高**: 解决了 Chameleon 全离散 token 在生成上的质量退化
4. **比 cascade 更端到端**: 不像 BAGEL 那样视觉部分 frozen，两塔联合优化

## 代表工作

- [[Cosmos3]]: AR (Qwen3-VL backbone) + Diffusion 双塔，统一 VLM / video gen / world sim / VLA 四类任务
- [[MoT]]: 早期提出的多模态 MoT 路由概念

## 关联

- [[MoT|Mixture-of-Transformers]]: 基础路由机制
- [[JointSelfAttention]]: 双塔间的信息融合方式
- [[Diffusion Model]] + [[Next-Token Prediction]]: 两塔分别的训练目标
