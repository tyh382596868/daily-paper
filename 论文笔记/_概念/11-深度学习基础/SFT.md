---
type: concept
aliases: [Supervised Fine-Tuning, 监督微调]
---

# SFT

## 定义

Supervised Fine-Tuning, 用标注好的输入-输出对在预训练模型上做监督微调,最大化条件似然 $\log p_\Theta(y \mid x)$。是 LLM/[[VLA]] 后训练的第一阶段, 通常先于 [[GRPO|RLVR]] / RLHF。

## 数学形式

$$
\mathcal{L}_{SFT}(\Theta) = -\mathbb{E}_{(x,y)\sim \mathcal{D}}\Big[\sum_{t=1}^{|y|} \log p_\Theta(y_t \mid x, y_{<t})\Big]
$$

## 核心要点

1. **梯度统计**: SFT 阶段[[梯度信噪比]]较高、梯度近似满秩, 适合 [[AdamW]] / [[Muon]] 等优化器。
2. **VLA 场景**: 用专家演示数据训练动作头,损失可为 [[L1 损失]] 或 [[Flow Matching]]。
3. **对照 RLVR**: 后接 [[GRPO]] 时, 梯度 SNR 显著下降, 需要 [[Pion]] 这类高通优化器。
4. **数据需求**: 高质量标注数据是瓶颈。

## 代表工作

- [[Pion]]: 在 SFT vs GRPO 的梯度 SNR 对比中作为参考阶段

## 相关概念

- [[GRPO]]
- [[预训练]]
- [[VLA]]
- [[梯度信噪比]]
