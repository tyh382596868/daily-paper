---
type: concept
aliases: [NTP, 下一 token 预测, Autoregressive Loss]
---

# Next-Token Prediction

## 定义

**Next-Token Prediction (NTP)** 是自回归语言模型与多模态 [[VLM]] / [[VLA]] 的核心训练目标：给定前 $i-1$ 个 token $x_{<i}$，预测第 $i$ 个 token 的概率分布 $p_\theta(x_i \mid x_{<i})$，损失为交叉熵：

$$
\mathcal{L}_\text{AR} = -\sum_i \log p_\theta(x_i \mid x_{<i})
$$

## 核心要点

1. **causal mask**: attention 中只能看到自己之前的 token
2. **离散 token 输出**: 通过 softmax 在词表上做分类
3. **scaling**: 是 LLM、VLM、Chameleon、[[Cosmos3]] AR 塔等的基础损失
4. **vs Diffusion**: 与 [[Flow Matching]] / 扩散损失互补，前者擅长离散符号、后者擅长连续信号

## 代表工作

- [[Cosmos3]]: AR 塔使用 NTP 训练文本 + 离散动作 token
- [[Qwen3-VL]]: VLM 的核心训练目标
- [[Chameleon]]: 所有模态 token 化后统一 NTP

## 关联

- [[Causal Self-Attention]]: NTP 必需的掩码方式
- [[Autoregressive]]: 同义概念
- [[Flow Matching]]: 互补的连续生成损失
