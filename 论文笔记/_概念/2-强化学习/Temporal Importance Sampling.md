---
type: concept
aliases: [TIS, Temporal Importance Sampling, 时序重要性采样]
---

# Temporal Importance Sampling (TIS)

## 定义

TIS 是 [[X-Foresight]] 提出的训练样本重加权策略：对每个时间 chunk，根据**加速度幅值**（横向 + 纵向）在三个时间窗口下的聚合分数计算重要性 $w_k$，再做温度缩放归一化得到采样概率 $p_k$。目的是让训练偏向"急刹 / 急转 / 紧急 cut-in"等安全敏感片段，缓解驾驶数据中长尾稀缺。

## 数学形式

**重要性分数**:
$$
w_k = \sum_{W \in \{W_1^k, W_2^k, W_3^k\}} \max_{t \in W} \left( \lambda_x |a_x(t)| + \lambda_y |a_y(t)| \right)
$$

**温度缩放采样**:
$$
p_k = \frac{w_k^{1/\tau}}{\sum_j w_j^{1/\tau}}
$$

## 三个时间窗口

- **$W_1^k$（near-future）**: 强调即将发生的事件——刹车、加速、突然变道起点
- **$W_2^k$（mid-horizon）**: 捕捉 maneuver commitment——开始转向、开始刹车
- **$W_3^k$（recent-history）**: 捕捉刚执行完的 maneuver 后效

## 核心要点

1. **基于物理量**: 用加速度而非人工标签，免标注
2. **温度可调**: $\tau$ 小→偏向高分；$\tau$ 大→趋向均匀
3. **横纵分别加权**: $\lambda_x, \lambda_y$ 控制纵 / 横加速度的相对权重
4. **与 [[Curriculum Learning with Extended Foresight|CLEF]] 互补**: CLEF 管时间结构，TIS 管样本分布
5. **在安全指标上贡献最大**: X-Foresight Table 2 中 TIS 单独把 collision rate 从 0.230% 降到 0.216%

## 代表工作

- [[X-Foresight]]: 首次提出，在 ablation 中显示对 collision rate 贡献最显著

## 相关概念

- [[Curriculum Learning with Extended Foresight]]
- [[课程学习]]
- [[Chunk-Wise Autoregression]]
