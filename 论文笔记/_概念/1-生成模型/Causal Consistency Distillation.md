---
type: concept
aliases: [因果一致性蒸馏, CCD]
---

# Causal Consistency Distillation

## 定义

把 [[Consistency Distillation]] 推广到**因果自回归**视频扩散场景：学生在 timestep $t$ 与 $t-\Delta t$ 上的预测在历史 chunk 条件下保持 endpoint 一致，从而把多步采样压缩到少步。

## 数学形式

$$
\mathcal{L}=\mathbb{E}\big[w(t)\,d\big(G_\theta(\boldsymbol{x}_t^i,\boldsymbol{x}_\text{gt}^{<i},t),\;G_{\theta^-}(\hat{\boldsymbol{x}}^i_{t-\Delta t},\boldsymbol{x}_\text{gt}^{<i},t-\Delta t)\big)\big]
$$

## 核心要点

1. **在线版**蒸馏，不需要预先生成 ODE 轨迹，节省存储和离线计算。
2. 历史 chunk $\boldsymbol{x}_\text{gt}^{<i}$ 作为条件喂给学生与 EMA target，保持因果性。
3. 早期可能不稳定，常与 warm-up 或 ODE 初始化配合。

## 代表工作

- [[minWM]]: Stage 2B 的核心算法，对应论文称为 [[Causal Forcing++]] 路径。

## 相关概念

- [[Consistency Distillation]]
- [[Causal Forcing]]
- [[Diffusion Forcing]]
