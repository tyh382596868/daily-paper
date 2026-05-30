---
type: concept
aliases: [因果 ODE 初始化, Causal ODE]
---

# Causal ODE Initialization

## 定义

[[Causal Forcing]] 的 Stage 2 训练目标：用双向 teacher 在每个 chunk 上离线跑出多步 ODE 解，把"含噪当前帧 + 真值历史"作为输入、clean 帧作为监督，回归训练少步学生 $G_\theta$。

## 数学形式

$$
\theta^{*}=\arg\min_{\theta}\mathbb{E}\big[\|G_{\theta}(\boldsymbol{x}_{t}^{i},\boldsymbol{x}_{\mathrm{gt}}^{<i},t)-\boldsymbol{x}_{0}^{i}\|^{2}\big]
$$

## 核心要点

1. 监督信号 **clean 真值帧** $\boldsymbol{x}_0^i$，训练目标简单稳定。
2. 需要**离线生成 ODE 轨迹**作为训练数据，存储与算力开销大。
3. 一般作为 [[Causal Forcing]] 三阶段中的 Stage 2A；[[Causal Forcing++]] 用 [[Causal Consistency Distillation]] 取代它。

## 代表工作

- [[minWM]]: Stage 2A 的选项，与在线一致性蒸馏并列。

## 相关概念

- [[Causal Forcing]]
- [[Causal Consistency Distillation]]
- [[Teacher Forcing]]
- [[少步蒸馏]]
