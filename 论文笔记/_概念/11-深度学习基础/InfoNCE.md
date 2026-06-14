---
type: concept
aliases: [InfoNCE Loss, Noise Contrastive Estimation, 对比噪声估计]
---

# InfoNCE

## 定义
InfoNCE（Information Noise Contrastive Estimation）是一种自监督对比学习目标函数，通过最大化正样本对的互信息下界来学习表示。

## 数学形式
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top f(x^+)/\tau)}{\exp(f(x)^\top f(x^+)/\tau) + \sum_{i=1}^{N}\exp(f(x)^\top f(x^-_i)/\tau)}\right]$$

其中 $\tau$ 为温度参数，$x^+$ 为正样本，$x^-_i$ 为负样本。

## 核心要点
1. 是互信息 $I(x; c)$ 的下界，最小化 InfoNCE 等价于最大化互信息
2. 负样本数量 $N$ 越大，估计越精确，但计算成本越高
3. 温度参数 $\tau$ 控制分布的尖锐程度：小 $\tau$ 使分布更集中
4. SimCLR、MoCo、CLIP 等均基于此目标函数的变体

## 代表工作
- [[SimCLR]]: 视觉对比学习经典应用
- [[MoCo]]: 动量对比学习
- [[CLIP]]: 跨模态对比学习

## 相关概念
- [[Contrastive Learning]]
- [[Self-Supervised Learning]]
