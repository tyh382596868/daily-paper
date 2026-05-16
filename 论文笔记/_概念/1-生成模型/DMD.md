---
type: concept
aliases: [DMD, Distribution Matching Distillation, 分布匹配蒸馏]
---

# DMD

## 定义

**Distribution Matching Distillation**——把多步扩散模型蒸馏成一步（或少步）生成器的方法。核心思想是让学生生成器输出的分布在 score function 层面对齐教师扩散模型，而不是逐点回归教师的预测。

## 数学形式

设教师扩散模型对应数据分布 $p_{\text{real}}$，学生 generator $G_\phi$ 诱导分布 $p_{\text{fake}}$。DMD 最小化二者在所有噪声水平上的 KL 散度。等价梯度为：

$$\nabla_\phi \mathcal{L}_{\text{DMD}} = \mathbb{E}_{t, z}\big[ (s_{\text{fake}}(z_t, t) - s_{\text{real}}(z_t, t)) \cdot \nabla_\phi G_\phi \big]$$

其中 $s_{\text{real}}$ 来自冻结的教师扩散模型，$s_{\text{fake}}$ 是一个跟随训练的辅助扩散模型，在生成分布上拟合 score。

## 核心要点

1. **一步生成**：和 [[Consistency Model]] 同属"扩散→一步"路线，但路径不同——CM 端点匹配，DMD 分布匹配
2. **双网络结构**：除了 generator $G_\phi$，还要训练一个 fake-score network 不断追踪生成分布
3. **DMD2**：去掉了 regression loss，用 two-time-scale update 让训练更稳，质量也更高
4. **常用于 video diffusion 蒸馏**：CausVid、RAVEN 等都基于 DMD 路线做因果视频生成
5. **缺点**：训练复杂，需要同时维护教师、学生、fake-score 三个网络

## 代表工作

- Yin et al. "One-step Diffusion with Distribution Matching Distillation" (DMD, 2023)
- DMD2 (2024)：去 regression、加 GAN-style loss
- CausVid：把 DMD 用到因果 video diffusion
- RAVEN (2026)：在 CM/DMD 蒸馏后再加 CM-GRPO 做 RL 微调

## 相关概念

- [[Consistency Model]]
- [[扩散变换器]]
- [[Flow Matching]]
- [[视频扩散模型]]
- [[GRPO]]
