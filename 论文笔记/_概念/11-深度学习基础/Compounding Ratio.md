---
type: concept
aliases: [复合比, CR, 累积误差比, 误差复合率]
---

# Compounding Ratio（复合比）

## 定义

Compounding Ratio（CR）是衡量自回归动力学模型在开环滚出时误差积累程度的指标。它是自由滚出（递归预测）误差与教师强制（Teacher Forcing）误差之比：CR 越接近 1，说明递归推演引入的额外误差越小，模型长时域稳定性越好。

## 数学形式

$$
\mathrm{CR}_k = \frac{e^{\mathbf{x}}_{k,\mathrm{rollout}}}{e^{\mathbf{x}}_{k,\mathrm{TF}}}
$$

其中归一化预测误差为：

$$
e^{\mathbf{x}}_k = \sqrt{\frac{1}{D_s}\|\tilde{\mathbf{x}}_k - \mathbf{x}_k\|_2^2}
$$

## 核心要点

1. **CR > 1**: 意味着自回归递归引入了额外误差积累；CR 越大，[[Compounding Errors|复合误差]]越严重
2. **CR = 1**: 理想情况，自由滚出与教师强制完全一致（无额外误差传播）
3. **CR < 1**: 极少见，通常意味着模型在自由运行时反而比单步更准（如有正则化效果）
4. **与 Teacher Forcing 的关系**: TF 给模型喂真实历史状态，消除误差传播；Rollout 用预测状态递归，暴露复合误差

## 代表工作

- [[SkyJEPA]]: 在 60 步滚出时 CR=1.4，显著优于自回归预测基线的 CR=2.4，证明了 [[JEPA]] 潜预测的稳定性优势

## 相关概念

- [[Compounding Errors]]: CR 量化的正是复合误差的严重程度
- [[Teacher Forcing]]: CR 的分母，单步有监督预测误差
- [[JEPA]]: SkyJEPA 用 JEPA 潜预测降低 CR 的核心方法
