---
type: concept
aliases: [Action Chunking with Transformers, ACT Policy, 动作分块变换器]
---

# ACT

## 定义
ACT（Action Chunking with Transformers）是一种用于机器人操作的模仿学习方法，通过预测动作序列块（chunk）而非单步动作来减少复合误差，是双臂操作（ALOHA 平台）的标准 baseline。

## 数学形式
条件 VAE 结构：encoder $q_\phi(z|a_{1:k}, o)$ 编码动作块，decoder $p_\theta(\hat{a}_{1:k} | z, o)$ 重建：

$$\mathcal{L}_{ACT} = \mathbb{E} \left[ \sum_{i=1}^{k} \| a_i - \hat{a}_i \|^2 \right] + \beta \cdot D_{KL}(q_\phi \| \mathcal{N}(0,I))$$

推理时 $z=0$（mean），直接解码 $k$ 步动作，通过时序集成（temporal ensembling）融合重叠预测。

## 核心要点
1. 动作分块（chunk size $k=100$）大幅降低 compounding error
2. CVAE 结构在训练时引入动作 latent，推理时 condition on $z=0$
3. 时序集成：每步预测的动作用指数加权平均融合
4. 是 ALOHA / 双臂操作的核心 baseline，被 EventVLA、Tri-Info 等大量引用

## 代表工作
- [[ALOHA]]: ACT 的首发平台，双臂灵巧操作
- [[EventVLA]]: 以 ACT 为 baseline 对比长程操作
- [[Tri-Info]]: 用 ACT policy 验证失败预测

## 相关概念
- [[ALOHA]]
- [[Diffusion Policy]]
- [[Imitation Learning]]
- [[Temporal Ensembling]]
