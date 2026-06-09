---
type: concept
aliases: [filtered behavior cloning, success-filtered BC]
---

# Filtered BC

## 定义
Behavior Cloning 的变体：只在被判定为「成功」的演示子集上训练，过滤掉失败或低质量的轨迹，以减少噪声监督信号的干扰。

## 数学形式
$$\mathcal{L}_{\text{Filtered BC}} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{success}}} \left[ \| \pi_\theta(s) - a \|^2 \right]$$

其中 $\mathcal{D}_{\text{success}} = \{(s,a) \in \mathcal{D} \mid \text{success}(\tau) = 1\}$。

## 核心要点
1. 相比标准 BC，降低了失败演示对训练的负面影响
2. 缺点：丢弃了部分成功轨迹中的有用子轨迹；数据利用率低
3. 「成功」标签的获取需要人工标注或自动检测，本身有成本
4. 在混合质量数据场景下，filtered BC 不如能统一利用所有数据的方法（如 FlowPRO/ForesightFlow）

## 代表工作
- [[ForesightFlow]]：将 filtered BC 作为基线，提出利用完整混合质量经验的替代方案
- [[FlowPRO]]：同样与 filtered BC 对比，证明偏好优化方法的优越性

## 相关概念
- [[behavior cloning]]（基础方法）
- [[VLA 后训练]]（常用场景）
- [[DAgger]]（另一种解决数据质量问题的方法）
