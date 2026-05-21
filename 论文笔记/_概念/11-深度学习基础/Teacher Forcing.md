---
type: concept
aliases: [教师强制, TF]
---

# Teacher Forcing

## 定义
训练自回归模型时，**用 ground-truth 上一步输出**而非模型自身预测作为下一步输入的训练策略。

## 数学形式
对于自回归预测 $\hat{y}_{t+1} = f(y_{1:t})$，训练损失为：
$$
\mathcal{L} = \sum_t \ell\big(f(y_{1:t}), y_{t+1}\big)
$$
（用真值 $y_{1:t}$ 而非 $\hat{y}_{1:t}$）

## 核心要点
1. **训练-推理分布不匹配**：推理时模型只能用自己的预测，但训练时只见过真值——导致 [[Compounding Errors]]
2. 缓解策略：
   - **多步 rollout 训练**（[[JEPA-WM]] 中的 K-step loss）
   - **Scheduled Sampling**（按概率混合真值与预测）
   - **DAgger / Imitation Learning** 的训练范式
3. 在世界模型中，纯 TF 训练（K=1）会导致开环预测快速发散

## 代表工作
- [[JEPA-WM]]: 系统分析 K-step rollout 训练替代 TF 的效果
- [[DreamerV3]]: 用 imagination rollout 缓解 TF 的训练-推理 gap

## 相关概念
- [[Compounding Errors]]
- [[JEPA-WM]]
- [[Lipschitz 常数]]
