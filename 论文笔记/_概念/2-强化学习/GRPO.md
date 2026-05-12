---
type: concept
aliases: [GRPO, Group Relative Policy Optimization, 组相对策略优化]
---

# GRPO（Group Relative Policy Optimization）

## 定义
一种去掉 value/critic 网络的策略梯度算法：对同一个 prompt/状态采样一组（group）回答，用组内相对优势（reward 减去组均值、再除以组标准差）作为 advantage 来更新策略。

## 数学形式
对一组 $G$ 个样本 $\{o_i\}$，其归一化优势为
$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$
目标函数与 PPO 类似，用 clipped surrogate + KL 正则约束新策略不偏离参考策略。

## 核心要点
1. 相比 [[PPO]] 省掉了单独的 value function——advantage 直接由组内相对排名给出，显存和实现都更轻。
2. 最初在 LLM 的 RLHF/推理训练里走红（DeepSeek 系），后被搬到 video diffusion、VLA 等的后训练。
3. 缺点：组采样成本高；对 reward 的尺度/方差敏感；组内样本多样性不足时优势信号会塌。

## 代表工作
- [[Sword]]: 用 GRPO 在学到的 world model（当模拟器）里对 [[OpenVLA]] 做 policy 后训练。

## 相关概念
- [[PPO]]
- [[Flow Matching]]
