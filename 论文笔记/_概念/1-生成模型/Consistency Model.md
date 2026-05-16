---
type: concept
aliases: [Consistency Model, CM, 一致性模型]
---

# Consistency Model

## 定义

一类把扩散模型轨迹中任意时间步 $t$ 的样本 $z_t$ 直接映射到原始干净样本 $z_0$ 的生成模型，从而实现 **一步或少步采样**，绕开标准扩散需要的几十到上千次去噪迭代。

## 数学形式

学习一个 consistency function $f_\theta(z_t, t)$ 满足边界条件 $f_\theta(z_0, 0) = z_0$，并对同一条 probability-flow ODE 轨迹上的任意两个时间步 $t, s$ 满足：

$$f_\theta(z_t, t) = f_\theta(z_s, s) = z_0$$

蒸馏目标（CD/CT）一般写作相邻两步的差异：

$$\mathcal{L} = \mathbb{E}\big[\| f_\theta(z_{t+\Delta}, t+\Delta) - f_{\theta^-}(z_t, t) \|\big]$$

其中 $\theta^-$ 是 EMA 教师参数。

## 核心要点

1. **一步采样**：直接 $z_T \to z_0$，比 [[扩散变换器]] 的多步采样快一到两个量级
2. **可叠加步数**：还能多步迭代换更高质量，但通常步数超过 4–8 后增益饱和甚至倒退（test-time scaling 问题）
3. **与 [[Flow Matching]] 的区别**：CM 端点匹配 $z_t \to z_0$；flow matching 学的是速度场 $v_\theta(z_t, t)$
4. **典型变体**：Consistency Distillation (CD) 从教师扩散模型蒸馏；Consistency Training (CT) 从头训
5. **后续改进**：[[DMD]]、Latent Consistency Model、Flow Map（如 AnyFlow）等都在试图解决 CM 多步退化问题

## 代表工作

- Song et al. "Consistency Models" (2023)：开山之作
- [[DMD]]：用 score distribution matching 改进 CM 多步质量
- RAVEN (2026)：在 CM 之上加 CM-GRPO 做 video diffusion 的 RL 微调
- AnyFlow (2026)：把 CM 的端点蒸馏改成任意区间的 flow map 蒸馏

## 相关概念

- [[扩散变换器]]
- [[Flow Matching]]
- [[DMD]]
- [[视频扩散模型]]
