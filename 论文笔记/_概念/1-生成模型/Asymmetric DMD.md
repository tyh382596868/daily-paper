---
type: concept
aliases: [Asymmetric Distribution Matching Distillation]
---

# Asymmetric DMD

## 定义

[[DMD]] 蒸馏的非对称变体：用**冻结的双向 teacher**作为 $s_\text{real}$、用**动态 fine-tune 的 fake score 网络**作为 $s_\text{fake}$，两者结构上不对称，专门用于把多步因果学生与高质量双向 teacher 的分布对齐。

## 数学形式

$$
\nabla_{\theta}\mathbb{E}_{t}\big[D_{\mathrm{KL}}\big(p_{\theta,t}\,\Vert\,p_{\text{data},t}\big)\big]=-\mathbb{E}\Big[\big(s_{\text{real}}-s_{\text{fake}}\big)\frac{\partial\tilde{\boldsymbol{x}}}{\partial\theta}\Big]
$$

## 核心要点

1. "Asymmetric" 体现在 $s_\text{real}$ 与 $s_\text{fake}$ 由不同模型（冻结 vs 在线）提供，避免对称蒸馏的退化。
2. 通常作为 AR 扩散蒸馏的**最后阶段**（数百步即可收敛），用于提质而非主训练。
3. 对相机条件等控制信号天然兼容：把同一条件喂给 real / fake 两个 score 网络即可。

## 代表工作

- [[minWM]]: 三阶段蒸馏 pipeline 的 Stage 3，单 A800 上把首帧延迟从 269s 压到 1.1s。

## 相关概念

- [[DMD]]
- [[Causal Forcing]]
- [[少步蒸馏]]
