---
type: concept
aliases: [CCI, 因果认知指数, Causality Cognition Index]
---

# Causality Cognition Index (CCI)

## 定义

由 [[YoCausal]] 提出的 Level 2 指标，通过比较 [[Video Diffusion Model|VDM]] 在因果子集 $\mathcal{D}_c$ 与非因果子集 $\mathcal{D}_{nc}$ 上的 [[Reverse Surprise Index|RSI]] 差异，把"真正的因果理解"从"浅层时间统计"里剥离出来。

## 数学形式

$$
\mathrm{CCI}(\mathcal{D}) = \mathrm{RSI}(\mathcal{D}_c) - \mathrm{RSI}(\mathcal{D}_{nc})
$$

归一化版（human = 100%）：

$$
\widetilde{\mathrm{CCI}}(\mathcal{D}) = \frac{\mathrm{CCI}(\mathcal{D})}{\mathrm{CCI}_{\mathrm{human}}(\mathcal{D})} \times 100\%
$$

## 核心要点

1. 正值：模型在因果视频上确实更"惊讶"于反向 — 有因果感
2. 负值：模型反而在非因果视频上更困惑 — 学到的是浅层时序统计
3. 因果分层由 [[Vision-Language Model|VLM]] (Gemini 3.0 Pro) 完成，与人工 Kendall $\tau=0.7613$
4. 与 [[Aesthetic Quality|美学质量]] 的 Kendall $\tau=0$，证明测的是正交能力
5. Human baseline = +8.67%，最强 VDM (Wan2.1-T2V-14B) = +5.91%（68% of human）

## 代表工作

- [[YoCausal]]: 提出 CCI，揭示"RSI 高但 CCI 几乎为 0"的现象（如 LTX-Video-2B）

## 相关概念

- [[Reverse Surprise Index]]
- [[Counterfactual]]
- [[Violation of Expectation]]
- [[VLM-as-Judge]]
- [[World Model]]
