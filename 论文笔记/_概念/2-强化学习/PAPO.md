---
type: concept
aliases: [PAPO, Planning-Aware Policy Optimization, 规划感知策略优化]
---

# PAPO（Planning-Aware Policy Optimization）

## 定义
一种规划感知的策略优化目标：在 [[GRPO]] 的 clipped surrogate 框架上，把优势替换为融合了[[规划动作]]因果重要性的[[规划感知优势]]，从而对关键规划动作重点优化。

## 数学形式
$$J_{\text{PAPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=0}^{T_i-1}\Big(\min\big(\rho_{i,t}\widehat{A}_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\widehat{A}_{i,t}\big) - \beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\Big)\right]$$
其中 $\widehat{A}_{i,t}$ 为规划感知优势，$\rho_{i,t}$ 为新旧策略重要性比率。

## 核心要点
1. 与 [[GRPO]] / [[PPO]] 同源，区别仅在于优势从组相对优势换成规划感知优势。
2. 保留 importance ratio 裁剪与对参考策略的 [[KL 散度]] 正则，保证后训练稳定。
3. 作为后训练算法可叠加在不同 VLA backbone 上，不改网络结构。

## 代表工作
- [[PAPO-VLA]]: 提出 PAPO 目标并用于 VLA 模型后训练。

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[规划感知优势]]
- [[KL 散度]]
