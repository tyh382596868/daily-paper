---
type: concept
aliases: [跨种子方差, inter-seed variance, 跨噪声种子方差]
---

# Inter-Seed Variance

## 定义

视觉[[世界模型]]幻觉检测信号之一：固定（上下文, 动作）输入对，使用 $N$ 个不同随机噪声种子独立去噪，预测结果的方差反映模型的认知不确定性，用于检测场景发散幻觉。

## 数学形式

$$
u_s = \mathrm{Var}_{k}[\hat{z}^{(k)}], \quad k = 1, \ldots, N
$$

运动归一化版本：

$$
u_s^{\text{norm}} = \frac{u_s}{m}
$$

## 核心要点

1. **认知不确定性代理**：训练数据充足时，不同种子收敛到同一预测；数据稀疏时，预测发散
2. **靶向场景发散幻觉**：高种子分歧区域正对应多步展开中产生物理不合理事件的区域
3. 需要多次前向推理（$N$ 次），计算代价高于 $u_r$ 和 $u_f$

## 代表工作

- [[MMBench2]]：与 $u_r$ 和 $u_f$ 共同验证 Spearman ρ ≈ 0.80

## 相关概念

- [[Round-Trip Residual]]
- [[Flow Instability]]
- [[Motion-Normalized Predictor]]
- [[世界模型]]
