---
type: concept
aliases: [Hybrid Terminal Cost, 混合终端代价]
---

# Hybrid Terminal Cost

## 定义

把原始 [[Terminal Proximity Cost|latent MSE]] 与学习型代价（如 [[TRM]] 头）**逐 batch 标准化后加权混合**，作为 [[Latent MPC]] 的终端代价。当学习型代价改善了排序但原始代价仍携带有用信号（接触、rollout 保真度等）时使用。

## 数学形式

$$
c_{\text{hyb}}^{(i)} = \frac{c_{\text{lat}}^{(i)} - \mu_{\text{lat}}}{\sigma_{\text{lat}} + \epsilon} + \lambda \cdot \frac{c_{\text{TRM}}^{(i)} - \mu_{\text{TRM}}}{\sigma_{\text{TRM}} + \epsilon}
$$

## 核心要点

1. **逐 batch 标准化是关键**: 避免两种代价的绝对尺度差异主导排序
2. **使用场景**: 替换模式失败但学习型代价 [[SCSA]] 排序显著优于原始代价时——典型如 [[PushT]] 接触类任务
3. **PushT go50 实证**: raw 40.0% → true hybrid 52.7%（shuffled hybrid 仅 42.7%）
4. **PushT go75 实证**: raw 16% → true hybrid 22%；SCSA selected-distance 191.6 → 114.5
5. **建议**: 当 raw proximity 是主要瓶颈用 replacement；当接触/rollout/recovery 仍限制成功率，用 hybrid 作为辅助

## 代表工作

- [[TRM]]: PushT 实验主推 hybrid 模式

## 相关概念

- [[Terminal Proximity Cost]]
- [[TRM]]
- [[Latent MPC]]
- [[SCSA]]
